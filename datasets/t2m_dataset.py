import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm.auto import tqdm
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.motion_process import recover_from_ric
from datasets.retrieval import *
# import clip
# import clip.model
from LongCLIP.model import longclip
from LongCLIP.model import model_longclip
import utils.paramUtil as paramUtil
from utils.plot_script import *
import json
from utils.motion_process import recover_from_ric
from utils.plot_script import draw_to_batch
import time
import os
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
class Text2MotionDataset(data.Dataset):
    """
    Dataset for Text2Motion generation task.
    """
    data_root = ''
    act_root_path = ''
    min_motion_len=40
    joints_num = None
    dim_pose = None
    max_motion_length = 320
    def __init__(self, opt, split, mode='train', accelerator=None, retrieval=None, prompt_path=None):
        self.max_text_len = getattr(opt, 'max_text_len', 20)
        self.unit_length = getattr(opt, 'unit_length', 4)
        self.split = split
        self.mode = mode
        self.clip_model = self.load_and_freeze_clip("/root/autodl-tmp/StableMoFusion/LongCLIP/checkpoints/longclip-B.pt").cuda()  ###
        # self.clip_model = self.load_and_freeze_clip("ViT-B/32").cuda()
        self.prompt_path = prompt_path
        self.re = retrieval
        motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        text_dir = pjoin(self.data_root, 'texts')
        self.re_motion_path = pjoin(self.act_root_path,'train','motion_re')
        self.act_data_root = pjoin(self.act_root_path,f'{split}')
        self.mean_std_path = pjoin(self.act_root_path,'train')
        self.feature_pkl_path = '/root/autodl-tmp/StableMoFusion/datasets/feature_pkl'

        if mode not in ['train', 'eval','gt_eval','xyz_gt','hml_gt']:
            raise ValueError(f"Mode '{mode}' is not supported. Please use one of: 'train', 'eval', 'gt_eval', 'xyz_gt','hml_gt'.")
        
        mean, std = None, None
        if mode == 'gt_eval':
            print(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
            # used by T2M models (including evaluators)
            mean = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_mean.npy'))
            std = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['eval']:
            print(pjoin(opt.meta_dir, 'std.npy'))
            # used by our models during inference
            mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
            std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        else:
            # used by our models during train
            mean = np.load(pjoin(self.data_root, 'Mean.npy'))
            std = np.load(pjoin(self.data_root, 'Std.npy'))
            # mean = np.load(pjoin(self.mean_std_path, 'Mean.npy'))
            # std = np.load(pjoin(self.mean_std_path, 'Std.npy'))
            
        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate ours norms to theirs
            self.mean_for_eval = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
        if mode in ['gt_eval','eval']:
            self.w_vectorizer = WordVectorizer(opt.glove_dir, 'our_vab')
        
        data_dict = {}
        # train_id_list = []  ##to store train id
        # train_file = pjoin(self.data_root, 'train.txt')
        # with cs.open(train_file, 'r') as f:
        #     for line in f.readlines():
        #         train_id_list.append(line.strip())
                
        id_list = []  ##to store id
        file = pjoin(self.data_root, f'{split}.txt')
        with cs.open(file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        caption_list_re = []
        motion_list_re = []
        # texts_feature_list  = []
        # text_motion_dict = {}
        
        # activity dataset initial
        txt_root = pjoin(self.act_data_root,'text')
        motion_root = pjoin(self.act_data_root,'motion')
        act_data_dict ={}
        for id in os.listdir(txt_root):
            motion = np.load(pjoin(motion_root,id.split('.')[0] + '.npy'))
            # if (len(motion))<self.min_motion_len or (len(motion)>=500):
            #     continue
            # text = []
            with cs.open(pjoin(txt_root, id.split('.')[0] + '.txt')) as f:
                for line in f.readlines():
                    text = line
            act_data_dict[id.split('.')[0]]={
                'motion':motion,
                'text':text,
                'length':len(motion)
            }
        
        for name in tqdm(id_list,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                # 读取出来的motion第一个维度是帧数，第二个维度263维包含以下信息1+2+1+63+126+66+4=263
                # 根部旋转速度：1维
                # 根部线速度：2维
                # 根部高度：1维
                # 关节相对位置数据（ric_data）：63维（21个关节，每个关节3维）
                # 关节旋转数据（rot_data）：126维（21个关节，每个关节6维）
                # 局部速度数据：66维（22个关节，每个关节3维）
                # 足部接触信息：4维
                if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict['caption'] = caption
                        
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            caption_list_re.append(caption) ###
                            motion_list_re.append(n_motion)
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    for tdict in text_data:
                        caprand = tdict['caption']
                        caption_list_re.append(caprand) ###
                        motion_list_re.append(motion)
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if mode=='train':
            if opt.dataset_name != 'amass':
                joints_num = self.joints_num
                # root_rot_velocity (B, seq_len, 1)
                std[0:1] = std[0:1] / opt.feat_bias
                # root_linear_velocity (B, seq_len, 2)
                std[1:3] = std[1:3] / opt.feat_bias
                # root_y (B, seq_len, 1)
                std[3:4] = std[3:4] / opt.feat_bias
                # ric_data (B, seq_len, (joint_num - 1)*3)
                std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
                # rot_data (B, seq_len, (joint_num - 1)*6)
                std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                            joints_num - 1) * 9] / 1.0
                # local_velocity (B, seq_len, joint_num*3)
                std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                        4 + (joints_num - 1) * 9: 4 + (
                                                                                                    joints_num - 1) * 9 + joints_num * 3] / 1.0
                # foot contact (B, seq_len, 4)
                std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                                4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

                assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            
            if accelerator is not None and accelerator.is_main_process:
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.act_data_dict = act_data_dict
        self.data_dict = data_dict
        self.captions_re = caption_list_re
        self.motions_re = motion_list_re
        self.lengths_re = [len(s) for s in caption_list_re]
        self.name_list = name_list
        
        
        if self.re:
            print("Initializing RetrievalDatabase......")
            # if not os.path.exists(pjoin(self.feature_pkl_path, 'feature.pickle')):
            #     self.texts_feature_list=[]
            #     for caption_re in tqdm(self.captions_re,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            #         text = longclip.tokenize(caption_re, truncate=True).cuda()
            #         feat_clip_text = self.clip_model.encode_text(text).float().cuda()
            #         self.texts_feature_list.append(feat_clip_text)
            #     with open(pjoin(self.feature_pkl_path, 'feature.pickle'), 'wb') as f:
            #         pickle.dump(self.texts_feature_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            # else:
            #     with open(pjoin(self.feature_pkl_path, 'feature.pickle'), 'rb') as f:
            #         self.texts_feature_list = pickle.load(f)
            # # self.texts_feature_list = np.array(self.texts_feature_list)
            # print(len(self.texts_feature_list),self.texts_feature_list[0])
            
            if not os.path.exists(pjoin(self.feature_pkl_path, 'feature_array.pickle')):
                self.texts_feature_list=[]
                for caption_re in tqdm(self.captions_re,disable=not accelerator.is_local_main_process if accelerator is not None else False):
                    text = longclip.tokenize(caption_re, truncate=True).cuda()
                    feat_clip_text = self.clip_model.encode_text(text).float().cuda()
                    # feat_clip_text = feat_clip_text.detach().cpu().numpy()
                    # print(feat_clip_text)
                    self.texts_feature_list.append(feat_clip_text.detach().cpu().numpy())
                with open(pjoin(self.feature_pkl_path, 'feature_array.pickle'), 'wb') as f:
                    pickle.dump(self.texts_feature_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(pjoin(self.feature_pkl_path, 'feature_array.pickle'), 'rb') as f:
                    self.texts_feature_list = pickle.load(f)
            # self.texts_feature_list = np.array(self.texts_feature_list)
            # print(len(self.texts_feature_list),self.texts_feature_list[0])
            
            # if not os.path.exists(pjoin(self.feature_pkl_path, 'feature_tensor.pickle')):
            #     self.texts_feature_list = torch.empty((len(self.captions_re), 1, 512), dtype=torch.float32).cuda()
            #     for i in tqdm(range(len(self.captions_re)),disable=not accelerator.is_local_main_process if accelerator is not None else False):
            #         text = longclip.tokenize(self.captions_re[i], truncate=True).cuda()
            #         # text = clip.tokenize(self.captions_re[i], truncate=True).cuda()
            #         feat_clip_text = self.clip_model.encode_text(text).float().cuda()
            #         self.texts_feature_list[i] = feat_clip_text
            #     with open(pjoin(self.feature_pkl_path, 'feature_tensor.pickle'), 'wb') as f:
            #         pickle.dump(self.texts_feature_list.cpu(), f, protocol=pickle.HIGHEST_PROTOCOL)
            # else:
            #     with open(pjoin(self.feature_pkl_path, 'feature_tensor.pickle'), 'rb') as f:
            #         self.texts_feature_list = pickle.load(f).cuda()
            # self.texts_feature_list = torch.tensor(self.texts_feature_list).cuda()
            
            self.retrieval = RetrievalDatabase(self.captions_re,self.motions_re,self.texts_feature_list,length=self.lengths_re,names = self.name_list, clip_model=self.clip_model).to('cuda')
            print("Finish Initializing RetrievalDatabase")
    
    def cal_sentence_length(self):
        plt.style.use('fivethirtyeight')
        self.captions_re_length = list(map(lambda x: len(x.split()), self.captions_re))
        print(self.captions_re_length[0],type(self.captions_re_length))
        count = Counter(self.captions_re_length)  
        df = pd.DataFrame.from_dict(count, orient='index', columns=['count'])  
        df.index.name = 'sentence_length'  
        df.reset_index(inplace=True)  
        plt.figure(figsize=(10, 6))  
        sns.barplot(x='sentence_length', y='count', data=df)
        plt.title('Sentence Length Distribution')  
        plt.xlabel('Sentence Length')  
        plt.ylabel('Count')
        plt.savefig('sentence_length_distribution.png')
        plt.show()

    def inv_transform(self, data):
        return data * self.std + self.mean

    def motion_fusion_concat(self,motion_list):
        motion_re = torch.cat(motion_list, dim=0)
        return motion_re
    
    def motion_linear_interpolate(self, motion1, motion2, num_transition_frames):
        last_frame_motion1 = motion1[-1]
        first_frame_motion2 = motion2[0]
        assert len(last_frame_motion1)==len(first_frame_motion2)
        transition_frames = np.linspace(last_frame_motion1.cpu().numpy(), first_frame_motion2.cpu().numpy(), num=num_transition_frames)
        concatenated_motion = torch.cat((motion1, torch.from_numpy(transition_frames).float().cuda(), motion2), dim=0).cuda()
        
        return concatenated_motion
    
    def motion_fusion_smooth_interpolate(self, motion_list, num_transition_frames):
        smooth_motion = motion_list[0]
        
        for i in range(1, len(motion_list)):
            smooth_motion = self.motion_linear_interpolate(torch.tensor(smooth_motion).cuda(), torch.tensor(motion_list[i]).cuda(), num_transition_frames).cuda()
        
        return smooth_motion
    
    def get_local_cap_from_caption(self,caption,prompt_path):
        with open(prompt_path, 'r') as f:  
            prompt = f.read()                           
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": caption}
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
        generated_text = response.choices[0].message.content
        lines = generated_text.split('\n')
        extracted_texts = []
        for line in lines:
            first_delim_pos = line.find('tuple_delimiter}"')
            if first_delim_pos != -1:
                second_delim_pos = line.find('tuple_delimiter}"', first_delim_pos + len('tuple_delimiter}"'))
                if second_delim_pos != -1:
                    start = second_delim_pos + len('tuple_delimiter}"')
                    end = line.find('"){record_delimiter', start)
                    if end != -1:
                        extracted_texts.append(line[start:end])
        return extracted_texts
            
    
    def get_motion_from_caption(self,caption,split):
        motion_re_list= []
        captions = self.get_local_cap_from_caption(caption, self.prompt_path)
        for cap in captions:  
            motion_retrieval = self.retrieval.retrieve(cap, split)
            # motion_re_list.append(torch.tensor(motion_retrieval).cuda())
            motion_re_list.append(motion_retrieval)
        self.motion_re_list = motion_re_list
        return self.motion_re_list
    
    def plot_motion(self, motion, name):
        pred_xyz = recover_from_ric(motion, 22).cuda()
        xyz = pred_xyz.reshape(1, -1, 22, 3).cuda()
        draw_to_batch(xyz.detach().cpu().numpy(),'a', [f'{str(name)}.gif'])
    
    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = longclip.load(clip_version, device='cuda')
        # clip_model, _ = clip.load(clip_version, device='cuda', jit=False)
        model_longclip.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Freeze CLIP weights
        # clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def __len__(self):
        return len(self.data_dict)
    
    def motion_normalization(self,motion_array):
        motion_array_normal = (motion_array - self.mean) / self.std
        return motion_array_normal

    def __getitem__(self, idx):
        # data = self.data_dict[self.name_list[idx]]
        idx = f"{idx:06d}"
        data = self.act_data_dict[idx]
        # motion, m_length, text_list = data['motion'], data['length'], data['text']
        motion, m_length, caption = data['motion'], data['length'], data['text'] # act dataset

        # Randomly select a caption
        # text_data = random.choice(text_list)
        # caption = text_data['caption']
        # caption = random.choice(caption)

        "Z Normalization"
        if self.mode not in['xyz_gt','hml_gt']:
            motion = (motion - self.mean) / self.std

        "crop motion"
        if self.mode in ['eval','gt_eval']:
            # Crop the motions in to times of 4, and introduce small variations
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
        elif m_length >= self.max_motion_length:           
            idx = random.randint(0, len(motion) - self.max_motion_length)
            motion = motion[idx: idx + self.max_motion_length]
            m_length = self.max_motion_length
        
        "pad motion"
        if m_length < self.max_motion_length:
            pass
            motion = np.concatenate([motion,
                                        np.zeros((self.max_motion_length - m_length, np.array(motion).shape[1]))
                                        ], axis=0)
        assert len(motion) == self.max_motion_length


        # if self.mode in ['gt_eval', 'eval']:
        #     "word embedding for text-to-motion evaluation"
        #     tokens = text_data['tokens']
        #     if len(tokens) < self.max_text_len:
        #         # pad with "unk"
        #         tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #         sent_len = len(tokens)
        #         tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        #     else:
        #         # crop
        #         tokens = tokens[:self.max_text_len]
        #         tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #         sent_len = len(tokens)
        #     pos_one_hots = []
        #     word_embeddings = []
        #     for token in tokens:
        #         word_emb, pos_oh = self.w_vectorizer[token]
        #         pos_one_hots.append(pos_oh[None, :])
        #         word_embeddings.append(word_emb[None, :])
        #     pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        #     word_embeddings = np.concatenate(word_embeddings, axis=0)
        #     return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        # elif self.mode in ['xyz_gt']:
        #     "Convert motion hml representation to skeleton points xyz"
        #     # 1. Use kn to get the keypoints position (the padding position after kn is all zero)
        #     motion = torch.from_numpy(motion).float()
        #     pred_joints = recover_from_ric(motion, self.joints_num)  # (nframe, njoints, 3)  

        #     # 2. Put on Floor (Y axis)
        #     floor_height = pred_joints.min(dim=0)[0].min(dim=0)[0][1]
        #     pred_joints[:, :, 1] -= floor_height
        #     return pred_joints
        
        if self.re:    
            if not os.path.exists(pjoin(self.re_motion_path,str(idx) + '.npy')):
                motion_re_list = self.get_motion_from_caption(caption,split = self.split)
                local_action_num = len(motion_re_list)
                motion_re = self.motion_fusion_smooth_interpolate(motion_re_list, num_transition_frames=int(15*2/local_action_num))
                # motion_length = len(motion_re)
                # if motion_length > frames_length:
                #     # idx = random.randint(0, motion_length - frames_length)
                #     idx = torch.randint(0, motion_length - frames_length, (1,), device='cuda').item()
                #     motion_re = motion_re[idx: idx + frames_length]
                # else:
                #     motion_re = torch.cat((motion_re,
                #                             torch.zeros((frames_length - motion_length, motion_re.shape[1])).cuda()), dim=0).cuda()
                # assert len(motion_re) == frames_length
                motion_re_array = motion_re.cpu().numpy()
                np.save(pjoin(self.re_motion_path,str(idx)+'.npy'),motion_re_array)
                motion_retrieval = self.motion_normalization(motion_re_array)
                return caption, motion, m_length, motion_retrieval
            else:
                motion_retrieval = self.motion_normalization(np.load(pjoin(self.re_motion_path,idx,'.npy')))
                return caption, motion, m_length, motion_retrieval
        else:
            return caption, motion, m_length
         
    def get_local_action_motions(self,accelerator):
        # lo_txt_root = pjoin(self.act_data_root,'local_text')
        lo_txt_root = '/root/autodl-tmp/StableMoFusion/act/local_text'
        list_lo_txt_root = os.listdir(lo_txt_root)
        for txt_id in tqdm(list_lo_txt_root,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            with cs.open(pjoin(lo_txt_root, txt_id)) as fi:
                file_list = fi.readlines()
                if list_lo_txt_root.index(txt_id)>=24546 and list_lo_txt_root.index(txt_id)<26076:
                    split_re = 'val'
                elif list_lo_txt_root.index(txt_id)>=26076:
                    split_re = 'test'
                else:
                    split_re = 'train'
                # print(file_list)
                motion_re_list=[]
                for cap in file_list:
                    motion_retrieval = self.retrieval.retrieve(cap, split = split_re)
                    motion_re_list.append(torch.from_numpy(motion_retrieval))
                    # local_action_num = len(motion_re_list)
                    motion_re = self.motion_fusion_smooth_interpolate(motion_re_list, num_transition_frames=10)
                    motion_re_array = motion_re.cpu().numpy()
                    np.save(pjoin('/root/autodl-tmp/StableMoFusion/act/motion_re/',f'{txt_id.split('.')[0]}.npy'),motion_re_array)
                    # np.save(pjoin(self.act_data_root,'local_action',f'{txt_id.split('.')[0]}-{file_list.index(cap)+1}.npy'),motion_retrieval)
                    # np.save(pjoin('/root/autodl-tmp/StableMoFusion/act/local_action',f'{txt_id.split('.')[0]}-{file_list.index(cap)+1}.npy'),motion_retrieval)
    
    def get_local_action_motions_humanml3d(self,accelerator):
        def get_split_id(split):
            split_id_list = [] 
            file = pjoin('/root/autodl-tmp/StableMoFusion/data/HumanML3D', f'{split}.txt')
            with cs.open(file, 'r') as f:
                for line in f.readlines():
                    split_id_list.append(line.strip())
            return split_id_list
        lo_txt_root = '/root/autodl-tmp/StableMoFusion/data/HumanML3D/local_text'
        list_lo_txt_root = os.listdir(lo_txt_root)
        for txt_id in tqdm(list_lo_txt_root,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            with cs.open(pjoin(lo_txt_root, txt_id)) as fi:
                file_list = fi.readlines()
                if txt_id.split('.')[0] in get_split_id('val'):
                    split_re = 'val'
                elif txt_id.split('.')[0] in get_split_id('test'):
                    split_re = 'test'
                else:
                    split_re = 'train'
                motion_re_list=[]
                for cap in file_list:
                    motion_retrieval = self.retrieval.retrieve(cap, split = split_re)
                    motion_re_list.append(torch.from_numpy(motion_retrieval))
                    # local_action_num = len(motion_re_list)
                    motion_re = self.motion_fusion_smooth_interpolate(motion_re_list, num_transition_frames=10)
                    motion_re_array = motion_re.cpu().numpy()
                    np.save(pjoin('/root/autodl-tmp/StableMoFusion/data/HumanML3D/motion_re/',f'{txt_id.split('.')[0]}.npy'),motion_re_array)    
    
    def test_pipeline(self,caption,num_transition_frames,frames_length):
        # print("len(self.name_list)=",len(self.name_list),self.name_list[:123])
        motion_re_list = self.get_motion_from_caption(caption)
        n = 1
        motion_re_list_tensor = []
        for mrl in motion_re_list:
            print('len(mrl)=',len(mrl))
            # self.plot_motion(torch.from_numpy(mrl).float().cuda(), n)
            motion_re_list_tensor.append(torch.from_numpy(mrl).float().cuda())
            n+=1
        # motion_re_list_tensor = torch.tensor(motion_re_list).float().cuda()
        motion_re_concat = self.motion_fusion_concat(motion_re_list_tensor)
        print('len(motion_re_concat)=',len(motion_re_concat))
        motion_re_smooth = self.motion_fusion_smooth_interpolate(motion_re_list_tensor, num_transition_frames).cuda()
        print('len(motion_re_smooth)=',len(motion_re_smooth))
        self.plot_motion(motion_re_smooth,'motion_re_smooth_before')
        motion_length = len(motion_re_smooth)
        if motion_length >= frames_length:
            # idx = random.randint(0, motion_length - frames_length)
            idx = torch.randint(0, motion_length - frames_length, (1,), device='cuda').item()
            motion_re_smooth = motion_re_smooth[idx: idx + frames_length]
        else:
            motion_re_smooth = torch.cat((motion_re_smooth,
                                    torch.zeros((frames_length - motion_length, motion_re_smooth.shape[1])).cuda()), dim=0).cuda()
        assert len(motion_re_smooth) == frames_length 
        self.plot_motion(motion_re_concat,f'motion_re_concat_{random.randint(0,1000)}')
        # self.plot_motion(motion_re_smooth,'motion_re_smooth_frames_length')
        self.plot_motion(motion_re_smooth,f'motion_re_smooth_{random.randint(0,1000)}')
        return caption, motion_re_smooth
    
    def create_activity_dataset(self,save_root_path,split,num_transition_frames,accelerator):
        
        text_save_path = pjoin(save_root_path, split, 'caption')
        motion_save_path = pjoin(save_root_path, split, 'motion')
        # gif_save_path = pjoin(save_root_path, split, 'gif')
        # if split =='train':
        #     all_motion_re_smooth_array = []
        num = 1
        # tqdm(range(len(self.name_list)),disable=not accelerator.is_local_main_process if accelerator is not None else False):
        for name_id in tqdm(range(len(self.name_list)),disable=not accelerator.is_local_main_process if accelerator is not None else False):
            try:
                # t1= time.time()
                name1 = self.name_list[name_id]
                next_name_id = name_id+1
                next_name_id = (name_id + 1) % len(self.name_list)
                name2 = self.name_list[next_name_id]
                data1 = self.data_dict[name1]
                data2 = self.data_dict[name2]
                motion1, text_list1 = data1['motion'], data1['text']
                motion2, text_list2 = data2['motion'], data2['text']
                # t2 = time.time()
                # Randomly select a caption
                caption_filename = f"{num:06d}.txt"
                output_caption_filename = os.path.join(text_save_path, caption_filename)
                # for text1 in text_list1:
                #     text1['caption']
                    # 遍历所有组合  
                # caption1 = random.choice(text_list1)['caption']
                # caption2 = random.choice(text_list2)['caption']
                # caption_filename = f"{num:06d}.txt"
                # output_caption_filename = os.path.join(text_save_path, caption_filename)
                # t3 = time.time()
                with open(output_caption_filename, 'w') as f: 
                    for text1 in text_list1:  
                        for text2 in text_list2:  
                            f.write(text1['caption'] + '\n')  
                            f.write(text2['caption'] + '\n')
                    # f.write(f'{caption1}\n{caption2}\n')
                # motion1 = torch.tensor((motion1 - self.mean) / self.std).float().cuda()
                # motion2 = torch.tensor((motion2 - self.mean) / self.std).float().cuda()
                # t4 = time.time()
                motion1 = torch.tensor(motion1).float().cuda()
                motion2 = torch.tensor(motion2).float().cuda()
                motion_re_smooth = self.motion_linear_interpolate(motion1,motion2, num_transition_frames)
                # motion_length = len(motion_re_smooth)
                # t5 = time.time()
                # if motion_length > frames_length:
                #     # idx = random.randint(0, motion_length - frames_length)
                #     idx = torch.randint(0, motion_length - frames_length, (1,), device='cuda').item()
                #     motion_re_smooth = motion_re_smooth[idx: idx + frames_length]
                # else:
                #     motion_re_smooth = torch.cat((motion_re_smooth,
                #                             torch.zeros((frames_length - motion_length, motion_re_smooth.shape[1])).cuda()), dim=0).cuda()
                # assert len(motion_re_smooth) == frames_length
                # t6 = time.time()  
                motion_re_smooth_array = motion_re_smooth.cpu().numpy()  
                # if split =='train':
                #     all_motion_re_smooth_array.append(motion_re_smooth_array)
                motion_filename = f"{num:06d}.npy"
                output_motion_filename = os.path.join(motion_save_path, motion_filename)
                # t7 = time.time()
                np.save(output_motion_filename, motion_re_smooth_array)
                # t8 = time.time()
                ## remove create gif to run faster
                # pred_xyz = recover_from_ric(torch.from_numpy(motion_re_smooth_array).float().cuda(), 22)
                # t9 = time.time()
                # xyz = pred_xyz.reshape(1, -1, 22, 3).detach().cpu().numpy()
                # gif_filename = f'{num}.gif'
                # output_gif_filename = os.path.join(gif_save_path, gif_filename)
                # draw_to_batch(xyz,title_batch=[f'{num}'], outname=[output_gif_filename])
                # t10 = time.time()
                # print(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7)
                num+=1
            except Exception as e:
                print(f"Create activity dataset failed: {e}")
        # if split =='train':
        #     mean_std_path = pjoin(self.act_root_path,'train')
        #     train_data_npy = np.concatenate(all_motion_re_smooth_array, axis=0)
        #     output_mean_filename = os.path.join(mean_std_path, 'Mean.npy')
        #     np.save(output_mean_filename, np.mean(train_data_npy, axis=0))
        #     output_std_filename = os.path.join(mean_std_path, 'Std.npy')
        #     np.save(output_std_filename, np.std(train_data_npy, axis=0))
        #     print("Mean and Std .npy have been created")
            

class HumanML3D(Text2MotionDataset):
    def __init__(self, opt, split="train", mode='train', accelerator=None, retrieval=None, prompt_path=None):
        self.data_root = './data/HumanML3D'
        self.act_root_path = './activity'
        self.min_motion_len = 40
        self.joints_num = 22
        self.dim_pose = 263
        self.max_motion_length = 196
        if accelerator:
            accelerator.print('\n Loading %s mode HumanML3D %s dataset ...' % (mode,split))
        else:
            print('\n Loading %s mode HumanML3D dataset ...' % mode)
        super(HumanML3D, self).__init__(opt, split, mode, accelerator, retrieval, prompt_path)
        

class KIT(Text2MotionDataset):
    def __init__(self, opt, split="train", mode='train', accelerator=None, retrieval=None, prompt_path=None):
        self.data_root = './data/KIT-ML'
        self.min_motion_len = 24
        self.joints_num = 21
        self.dim_pose = 251
        self.max_motion_length = 196
        if accelerator:
            accelerator.print('\n Loading %s mode KIT %s dataset ...' % (mode,split))
        else:
            print('\n Loading %s mode KIT dataset ...' % mode)
        super(KIT, self).__init__(opt, split, mode, accelerator, retrieval, prompt_path)


        
            

