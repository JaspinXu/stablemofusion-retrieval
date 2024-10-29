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
        
    
        caption_list_re = []
        
        act_data_dict ={}
        lo_text_area = '/root/autodl-tmp/StableMoFusion/data/HumanML3D/local_text'
        list_lo_text_area = os.listdir(lo_text_area)
        
        for name in tqdm(list_lo_text_area,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            with cs.open(pjoin(lo_text_area, name)) as f:
                for line in f.readlines():
                    caption_list_re.append(line)

        self.captions_re = caption_list_re
        
        
        if self.re:
            print("Initializing RetrievalDatabase......")
            
            if not os.path.exists(pjoin(self.feature_pkl_path, 'feature_new_meta.pickle')):
                self.texts_feature_list=[]
                for caption_re in tqdm(self.captions_re,disable=not accelerator.is_local_main_process if accelerator is not None else False):
                    text = longclip.tokenize(caption_re, truncate=True).cuda()
                    feat_clip_text = self.clip_model.encode_text(text).float().cuda()
                    # feat_clip_text = feat_clip_text.detach().cpu().numpy()
                    # print(feat_clip_text)
                    self.texts_feature_list.append(feat_clip_text.detach().cpu().numpy())
                with open(pjoin(self.feature_pkl_path, 'feature_new_meta.pickle'), 'wb') as f:
                    pickle.dump(self.texts_feature_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(pjoin(self.feature_pkl_path, 'feature_new_meta.pickle'), 'rb') as f:
                    self.texts_feature_list = pickle.load(f)
            
            self.retrieval = RetrievalDatabase(self.captions_re,None,self.texts_feature_list,None,None, clip_model=self.clip_model).to('cuda')
            print("Finish Initializing RetrievalDatabase")
    
         
    def get_act_local_text_from_humanml3d(self,accelerator):
        lo_txt_root = '/root/autodl-tmp/StableMoFusion/act/local_text_nofilter'
        lo_txt_root_new = '/root/autodl-tmp/StableMoFusion/act/local_text'
        list_lo_txt_root = os.listdir(lo_txt_root)
        for txt_id in tqdm(list_lo_txt_root,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            new_local_text_list=[]
            with cs.open(pjoin(lo_txt_root, txt_id)) as fi:
                file_list = fi.readlines()
                for cap in file_list:
                    text_retrieval = self.retrieval.retrieve_text(cap)
                    new_local_text_list.append(text_retrieval)
            new_local_text_list_filter = self.remove_duplicates_except_first(new_local_text_list)
            output_caption_filename = pjoin(lo_txt_root_new, txt_id)
            with open(output_caption_filename, 'w', encoding='utf-8') as file:  
                for line in new_local_text_list_filter:  
                    file.write(line)
                    
    def remove_duplicates_except_first(self,lst):  
        seen = set()  
        result = []  
        for item in lst:  
            if item not in seen:  
                seen.add(item)  
                result.append(item)  
        return result
    
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


        
            

