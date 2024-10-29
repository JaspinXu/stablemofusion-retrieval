import sys
import os
from os.path import join as pjoin

from options.train_options import TrainOptions
from utils.plot_script import *

from models import build_models
from utils.ema import ExponentialMovingAverage
from trainers import DDPMTrainer
from motion_loader import get_dataset_loader

from accelerate.utils import set_seed
from accelerate import Accelerator
import torch
# import clip
# import clip.model
from torch import nn
import time

from datasets import get_dataset

from tqdm.auto import tqdm

if __name__ == '__main__':
     
    prompt_path ="/root/autodl-tmp/StableMoFusion/datasets/prompt/prompt.txt"
    
    accelerator = Accelerator()
    
    parser = TrainOptions()
    opt = parser.parse(accelerator)
    set_seed(opt.seed)
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if accelerator.is_main_process:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)

    dataset = get_dataset(opt, split='test', mode='train', accelerator=accelerator, retrieval=True, prompt_path=prompt_path)
    
    ########### test __getitem__ ###############
    # caption, motion, m_length, motion_retrieval = dataset.__getitem__(1)
    # print(caption, motion, m_length, motion_retrieval)
    ############################################
    
    # dataset.__getitem__(1)
    
    # for id in tqdm(range(1,24547),disable=not accelerator.is_local_main_process if accelerator is not None else False):
    #     dataset.__getitem__(id)
    # start = time.time()
    # dataset.get_local_action_motions(accelerator=accelerator)
    dataset.cal_sentence_length()
    # end = time.time()
    # print(end - start)
        
    
    ########### test pipeline ##################
    # 'A person walks forward while swinging their arms then plays a basketball',
    #                 'A man jumps a little then runs quickly then shakes their right hip',
    # test_caption = [
    #                 # 'A person paces back and forth and grabs something with their right hand',
    #                 'A person sits down and gestures out with their right arm 3 times',
    #                 'A person twists their body to the right and left then dances',
    #                 'A person kicks on their left leg then shrugs both arms']
    # for cap in test_caption:
    #     dataset.test_pipeline(cap,num_transition_frames= 15,frames_length=320)
    ############################################
    
    ### wrong method
    # dataset.init_database_and_retrieval_json(out_local_text_path, json_path, prompt_path)
    # dataset.init_database_and_retrieval_json_from_dataset(json_path,accelerator=accelerator)
    
    ### run method
    # accelerate launch --config_file 1gpu.yaml --gpu_ids 0 -m scripts.train_and_retrieval --name t2m_condunet1d --model-ema --dataset_name t2m
    # python -m scripts.generate --text_prompt "A person walks forward while swinging their arms then plays a basketball." --motion_length 4 --footskate_cleanup
    # python -m debugpy --listen 5678 --wait-for-client ./scripts/train_and_retrieval.py
    ### original method
    # train_datasetloader = get_dataset_loader(opt,  batch_size = opt.batch_size, split='train', accelerator=accelerator, mode='train', retrieval=True) 

#####

    # accelerator.print('\nInitializing model ...' )
    # encoder = build_models(opt)
    # model_ema = None
    # if opt.model_ema:
    #     # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
    #     # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
    #     adjust = 106_667 * opt.model_ema_steps / opt.num_train_steps
    #     alpha = 1.0 - opt.model_ema_decay
    #     alpha = min(1.0, alpha * adjust)
    #     print('EMA alpha:',alpha)
    #     model_ema = ExponentialMovingAverage(encoder, decay=1.0 - alpha)
    # accelerator.print('Finish building Model.\n')

    # trainer = DDPMTrainer(opt, encoder,accelerator, model_ema)

    # trainer.train(train_datasetloader)


