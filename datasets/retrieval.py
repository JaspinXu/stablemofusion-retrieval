from utils.plot_script import *
import torch
# import clip
# import clip.model
from LongCLIP.model import longclip
from torch import nn
import json
import os
import openai
from openai import OpenAI
import time
from tqdm.auto import tqdm

client = OpenAI(api_key='')
# export OPENAI_API_KEY='your_api_key_here'

class RetrievalDatabase(nn.Module):

    def __init__(self,
                captions,
                motions,
                text_features,
                length,
                names,
                num_retrieval=1,
                topk=None,
                retrieval_file=None,
                latent_dim=512,
                output_dim=512,
                num_layers=2,
                num_motion_layers=4,
                kinematic_coef=0.1,
                max_seq_len=196,
                num_heads=8,
                ff_size=1024,
                stride=4,
                sa_block_cfg=None,
                ffn_cfg=None,
                dropout=0,
                clip_model=None):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        self.num_layers = num_layers
        self.num_motion_layers = num_motion_layers
        self.max_seq_len = max_seq_len
        # self.captions=captions['caption']
        self.captions=captions
        self.motions=motions
        self.text_features=text_features
        self.length =length
        self.names = names
        self.clip_model =clip_model.cuda()
        # self.text_path =text_path
        # self.json_path = json_path
        # self.clip_model = self.load_and_freeze_clip("ViT-B/32")
        # with open('/CV/xhr/xhr_2/T2M-GPT/caption.json', 'r') as json_file:
        #     self.captions = json.load(json_file)['caption']
        # self.motions = np.load('/CV/xhr/xhr_2/T2M-GPT/motion.npy',allow_pickle=True)
        # # self.m_lengths = np.load('/CV/xhr/xhr_2/T2M-GPT/m_tokens_len.npy')
        # self.text_features=np.load('/CV/xhr/xhr_2/T2M-GPT/texts_feature.npy',allow_pickle=True)

        TransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu")
        self.text_encoder = nn.TransformerEncoder(
            TransEncoderLayer,
            num_layers=num_layers)
        self.results = {}

    def extract_text_feature(self, text, clip_model):
        text = longclip.tokenize([text], truncate=True).cuda()
        # text = clip.tokenize([text], truncate=True).cuda()
        with torch.no_grad():
            text_features = clip_model.encode_text(text).float().cuda()
        return text_features

    # def encode_text(self, text, device):
    #     with torch.no_grad():
    #         text = clip.tokenize(text, truncate=True).to(device)
    #         x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

    #         x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
    #         x = x.permute(1, 0, 2)  # NLD -> LND
    #         x = self.clip_model.transformer(x)
    #         x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

    #     # B, T, D
    #     xf_out = x.permute(1, 0, 2)
    #     return xf_out

    # def calculate(self,l_caption,l,lamb=0.1):
    #     return abs(l_caption-l)/max(l_caption,l)

    def get_local_text_from_text(self,out_local_text_path,prompt_path):
        
        try:
            with open(prompt_path, 'r') as f:  
                prompt = f.read()
            os.makedirs(out_local_text_path, exist_ok=True) 
            
            num = 1
            for caps in self.captions:                    
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": caps}
                    ]
                for attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=1024,
                        )
                        break
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed: {e}")
                        if attempt < 2:
                            time.sleep(5)
                        else:
                            raise  
                
                filename = str(num) +'.txt'

                # 提取所需内容
                generated_text = response.choices[0].message.content
                lines = generated_text.split('\n')
                extracted_texts = []
                for line in lines:
                    if 'A person' in line:
                        start = line.find('A person')
                        end = line.find('")', start)
                        if end != -1:
                            extracted_texts.append(line[start:end])

                # 保存提取的内容到文件
                output_filename = os.path.join(out_local_text_path, filename)
                with open(output_filename, 'w') as f:
                    for text in extracted_texts:
                        f.write(text + '\n')

                print(f"Processed {filename} and saved output to {output_filename}")
                num += 1
                
            assert num == len(self.captions),"num error"
            
            return True

        except:

            print(f"Failed to generate local text from llm")  
            return False

    def retrieval_json_from_text(self, out_local_text_path, json_path, prompt_path):
        # if self.get_local_text_from_text(out_local_text_path,prompt_path):
            # assert isinstance(local_text, list), "Local_text is not a list"
        local_text_motion_id_dict ={}
        # for local_cap in local_text:
        for filename in os.listdir(out_local_text_path):  
            if filename.endswith('.txt'):   
                with open(os.path.join(out_local_text_path, filename), 'r') as f:  
                    local_cap = f.read() 
                    local_lines = local_cap.split('\n')
                    for line in local_lines:
                        local_motion = self.retrieve(line,'cuda')
                        local_text_motion_id_dict[line] = local_motion
        directory = os.path.dirname(json_path)  
        if not os.path.exists(directory):  
            os.makedirs(directory) 
        print("Retrievaling motion id from local text...")
        with open(json_path, 'w', encoding='utf-8') as json_file:  ## json_path is a path including .json file
            json.dump(local_text_motion_id_dict, json_file, ensure_ascii=False, indent=4) 
        print("The json file has been created in {json_path}")
        # else:
        #     print("Generate retrieval json file failed")         
        
    def retrieval_json_from_dataset(self, json_path, accelerator):
        text_motion_dict ={}
        print("Retrievaling json from dataset...")
        try:
            for caps in tqdm(self.captions,disable=not accelerator.is_local_main_process if accelerator is not None else False):
                local_motion = self.retrieve(caps,'cuda')
                text_motion_dict[caps] = local_motion
            directory = os.path.dirname(json_path)  
            if not os.path.exists(directory):  
                os.makedirs(directory) 
            with open(json_path, 'w', encoding='utf-8') as json_file:  ## json_path is a path including .json file
                json.dump(text_motion_dict, json_file, ensure_ascii=False, indent=4) 
            print("The json file has been created in {json_path}")
        except Exception as e:
            print(f"Retrieval json failed: {e}")  


    # def retrieve(self, caption, split):### 106.39320397377014s for 000003.txt
    #     text_feature = self.extract_text_feature(caption, self.clip_model).cuda()
    #     semantic_score=[]
    #     sim_score=[]
    #     for  i, feature_base in enumerate(self.text_features):
    #         sem_s=torch.nn.functional.cosine_similarity(feature_base.reshape(1,-1), text_feature.reshape(1,-1)).cuda()
    #         semantic_score.append(sem_s.cuda())
    #         length_score=sem_s.cuda()*torch.exp(-torch.tensor(0.1).cuda()*self.calculate(len(caption),len(self.captions[i])).cuda()).cuda()
    #         sim_score.append(length_score.cuda())
    #     score=torch.tensor(sim_score).cuda()

    #     indexes = torch.argsort(score, descending=True).cuda()
        
    #     if split == 'train':
    #         # avoid same caption
    #         for index in indexes:
    #             if self.captions[index] != caption:
    #                 motion_re = self.motions[index]
    #                 return motion_re
    #     elif split in ['val','test']:
    #         motion_re = self.motions[indexes[0]]
    #         return motion_re
    #     else:
    #         print("split is a wrong parameter!")
    
    def cal_cosine_similarity(self,feature_base_np,text_feature_np):
        dot_product = np.dot(feature_base_np, text_feature_np.T) 
        norm_feature_base = np.linalg.norm(feature_base_np)  
        norm_text_feature = np.linalg.norm(text_feature_np)   
        cosine_similarity = dot_product / (norm_feature_base * norm_text_feature + 1e-9)
        return cosine_similarity
            
    # def retrieve(self, caption, split):  ### 90.79858613014221s for 000003.txt
    #     text_feature = self.extract_text_feature(caption, self.clip_model).cuda()
    #     batch_size = len(self.text_features)  
    #     semantic_scores = torch.empty(batch_size, dtype=torch.float32, device='cuda')   
    #     sim_scores = torch.empty(batch_size, dtype=torch.float32, device='cuda')   
    #     for i, feature_base in enumerate(self.text_features):
    #         sem_s = torch.nn.functional.cosine_similarity(feature_base, text_feature, dim=1).cuda()  
    #         semantic_scores[i] = sem_s
    #         length_score = sem_s * math.exp(-0.1* self.calculate(len(caption),len(self.captions[i])))
    #         sim_scores[i] = length_score
    #     indexes = torch.argsort(sim_scores, descending=True)          
    #     if split == 'train':  
    #         for index in indexes:  
    #             if self.captions[index.item()] != caption:  
    #                 return self.motions[index.item()]  
    #     elif split in ['val', 'test']:  
    #         return self.motions[indexes[0].item()]  
    #     else:  
    #         print("split is a wrong parameter!")  
    #         return None
    
    # def retrieve(self, caption, split): 
    #     text_feature = self.extract_text_feature(caption, self.clip_model).cuda()
    #     text_feature_array = text_feature.detach().cpu().numpy()  
    #     semantic_score=[]
    #     sim_score=[]
    #     for i, feature_base in enumerate(self.text_features):
    #         sem_s = self.cal_cosine_similarity(feature_base,text_feature_array)
    #         semantic_score.append(sem_s)
    #         length_score = sem_s * math.exp(-0.1* self.calculate(len(caption),len(self.captions[i])))
    #         sim_score.append(length_score[0][0]) 
    #     indexes = np.argsort(np.array(sim_score)).tolist()
    #     indexes = indexes[::-1]     
    #     if split == 'train':  
    #         for index in indexes: 
    #             if self.captions[index] != caption: 
    #                 print(self.captions[index]) 
    #                 return self.motions[index]  
    #     elif split in ['val', 'test']:  
    #         return self.motions[indexes[0]]  
    #     else:  
    #         print("split is a wrong parameter!")  
    #         return None
    def process_sentence(self,sentence):
        words = sentence.split()
        new_words = [word for word in words if word.lower() not in ['left', 'right','left.','right.']]
        return " ".join(new_words) if new_words else sentence
        
    def retrieve(self, caption, split): 
    
        text_feature = self.extract_text_feature(caption, self.clip_model).cuda()  # Shape: (feature_dim,)
        
        
        if not hasattr(self, 'text_features_tensor'):
            
            self.text_features_tensor = torch.stack([torch.from_numpy(f) if isinstance(f, np.ndarray) else f
                                                    for f in self.text_features]).cuda()  # Shape: (num_features, feature_dim)
        
        
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)  # Shape: (feature_dim,)
        text_features_norm = self.text_features_tensor / self.text_features_tensor.norm(dim=-1, keepdim=True)  # Shape: (num_features, feature_dim)
        
        text_features_norm = text_features_norm.squeeze(1)
        sem_s = torch.matmul(text_features_norm, text_feature_norm.T)  #
        sem_s = sem_s.squeeze(1)
        len_caption = len(caption)
        caption_lens = torch.tensor([len(c) for c in self.captions], device='cuda')  
        len_diffs = self.calculate(len_caption, caption_lens)  
        length_score = sem_s * torch.exp(-0.1 * len_diffs) 
        del text_feature_norm
        del text_features_norm
        del sem_s
        del len_diffs
        del caption_lens
        torch.cuda.empty_cache()
        k = 100  # 需要的元素数量
        # 获取前 k 个最小值的索引
        indexes = torch.topk(length_score, k, largest=True).indices.cpu().tolist()
        if split == 'train':  
            for index in indexes:
                index = int(index) 
                if self.captions[index] != caption and self.process_sentence(self.captions[index])!=self.process_sentence(caption):
                        return self.motions[index]
        elif split in ['val', 'test']:
            index = int(indexes[0])
            return self.motions[index]
        else:
            print("split is a wrong parameter!")
            return None
        
    def calculate(self, len_caption, caption_lens):
        diff = (len_caption - caption_lens).float()
        return torch.abs(diff)
    
    def retrieve_text(self, caption): 
    
        text_feature = self.extract_text_feature(caption, self.clip_model).cuda()  # Shape: (feature_dim,)
        
        
        if not hasattr(self, 'text_features_tensor'):
            
            self.text_features_tensor = torch.stack([torch.from_numpy(f) if isinstance(f, np.ndarray) else f
                                                    for f in self.text_features]).cuda()  # Shape: (num_features, feature_dim)
        
        
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)  # Shape: (feature_dim,)
        text_features_norm = self.text_features_tensor / self.text_features_tensor.norm(dim=-1, keepdim=True)  # Shape: (num_features, feature_dim)
        
        text_features_norm = text_features_norm.squeeze(1)
        sem_s = torch.matmul(text_features_norm, text_feature_norm.T)  #
        sem_s = sem_s.squeeze(1)
        len_caption = len(caption)
        caption_lens = torch.tensor([len(c) for c in self.captions], device='cuda')  
        len_diffs = self.calculate(len_caption, caption_lens)  
        length_score = sem_s * torch.exp(-0.1 * len_diffs) 
        del text_feature_norm
        del text_features_norm
        del sem_s
        del len_diffs
        del caption_lens
        torch.cuda.empty_cache()
        k = 100  # 需要的元素数量
        # 获取前 k 个最小值的索引
        indexes = torch.topk(length_score, k, largest=True).indices.cpu().tolist()
        for index in indexes:
            index = int(index) 
            if self.process_sentence(self.captions[index])!=self.process_sentence(caption):
                    return self.captions[index]
  



    # def generate_src_mask(self, T, length):
    #     B = len(length)
    #     src_mask = torch.ones(B, T)
    #     for i in range(B):
    #         for j in range(length[i], T):
    #             src_mask[i, j] = 0
    #     return src_mask

    # def forward(self, captions, clip_model, device, idx=None):
    #     B = len(captions)
    #     all_indexes = []
    #     for b_ix in range(B):
    #         # length = int(lengths[b_ix])
    #         if idx is None:
    #             batch_indexes = self.retrieve(captions[b_ix], clip_model, device)
    #         else:
    #             batch_indexes = self.retrieve(captions[b_ix], clip_model, device, idx[b_ix])
    #         all_indexes.extend(batch_indexes)
    #     all_indexes = np.array(all_indexes)
    #     N = all_indexes.shape[0]
    #     all_motions = torch.Tensor(self.motions[all_indexes]).to(device)
    #     all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long()
    #     all_captions = self.captions[all_indexes].tolist() 

    # def load_and_freeze_clip(self, clip_version):
    #     clip_model, _ = clip.load(  # clip_model.dtype=float32
    #         clip_version, device='cpu',
    #         jit=False)  # Must set jit=False for training
    #     clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    #     # Freeze CLIP weights
    #     clip_model.eval()
    #     for p in clip_model.parameters():
    #         p.requires_grad = False

    #     return clip_model
