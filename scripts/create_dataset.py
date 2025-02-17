import sys
import os
import spacy
from os.path import join as pjoin
from options.train_options import TrainOptions
from utils.plot_script import *
from accelerate.utils import set_seed
from accelerate import Accelerator
import torch
from torch import nn
from openai import OpenAI
import time
from tqdm.auto import tqdm
from datasets import get_dataset
import random

client = OpenAI(api_key='')

nlp = spacy.load("en_core_web_sm")

if __name__ == '__main__':
    
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
    
    def get_fusion_caption_from_cap(cap_path,fusion_prompt_path):
        with open(fusion_prompt_path, 'r') as f:  
            prompt = f.read()
        cap_files = [f for f in os.listdir(cap_path) if f.endswith('.txt')]
        print("Fusing caption......")
        for cap_file in tqdm(cap_files):
            try:
                cap_file_path = os.path.join(cap_path, cap_file)
                content_text =[]
                with open(cap_file_path, 'r') as f:
                    # for line in f.readlines():
                    caps = []
                    for cap in f.readlines():
                        caps.append(cap)
                        if len(caps) == 2:
                    #         process_lines(lines)
                    #         caps = []
                    # caps = f.read()                      
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
                            generated_text = response.choices[0].message.content
                            delim_pos = generated_text.find('tuple_delimiter}"')
                            if delim_pos != -1:
                                start = delim_pos + len('tuple_delimiter}"')
                                end = generated_text.find('"){record_delimiter', start)
                                if end != -1:
                                    # generated_text[start:end]
                                    content_text.append(str(generated_text[start:end]) + '\n')
                                    caps = []
                with open(cap_file_path, 'w') as f:
                    f.write(f'{content_text}')
            except Exception as e:
                print(f"Fusion caption failed: {e}")
            
    nlp = spacy.load("en_core_web_sm")

    # def extract_subject_and_action(sentence):
    #     doc = nlp(sentence)
    #     subject = None
    #     actions = []

    #     for token in doc:
    #         # 提取主语
    #         if "subj" in token.dep_:
    #             subject = token.text
            
    #         # 提取动词及其后面的描述
    #         if token.dep_ == "ROOT":
    #             action_phrase = []
    #             for child in token.children:
    #                 action_phrase.append(child.text)
    #             action_phrase = ' '.join(action_phrase).strip()
    #             actions.append(token.text + ' ' + action_phrase)

    #     return subject, actions
    
    # def extract_subject_and_action(sentence):
    #     doc = nlp(sentence)
    #     subject = None
    #     action_sentence = None

    #     for token in doc:
    #         # 提取主语
    #         if "subj" in token.dep_:
    #             subject = token.text
    #             # 从主语后提取整个句子
    #             action_sentence = ' '.join([t.text for t in doc[token.i:]]).strip()  # 从主语开始到句尾
    #             action_sentence = action_sentence.rstrip('.')  # 去除句尾的句号
    #             break  # 找到主语后退出循环

    #     return subject, action_sentence
    
    def extract_subject_and_action(sentence):
        doc = nlp(sentence)
        subject = ''
        before_subject = ""
        after_subject = ""

        for i, token in enumerate(doc):
            # 提取主语
            if "subj" in token.dep_:
                subject = str(token.text)
                
                # 提取主语前的内容
                before_subject = ' '.join([t.text for t in doc[:i]]).strip()
                # 提取主语后的内容
                after_subject = ' '.join([t.text for t in doc[i + 1:]]).strip()
                after_subject = after_subject.rstrip('.')  # 去除句尾的句号
                break  # 找到主语后退出循环

        return [str(before_subject + ' ' + subject).strip()], [str(after_subject).strip()]


    def process_txt_files(directory):
        connectives = [
            "then",
            "next",
            "after that",
            "subsequently",
            "later",
            "thereafter"
        ]
        for filename in os.listdir(pjoin(directory,'caption')):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(directory,'caption', filename)
                output_file_path = os.path.join(directory, 'text', filename)
                
                with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
                    lines = infile.readlines()
                    
                    combined_sentences = []
                    for i in range(0, len(lines), 2):
                        first_line = lines[i].strip()
                        if i + 1 < len(lines):
                            second_line = lines[i + 1].strip()

                            subject1, actions1 = extract_subject_and_action(first_line)
                            _, actions2 = extract_subject_and_action(second_line)
                            
                            connective = random.choice(connectives)
                            if subject1 and len(subject1) > 0:
                                subject1 = subject1[0]
                            else:
                                subject1 = ""
                            if actions1 and len(actions1) > 0:
                                action1 = actions1[0]
                            else:
                                action1 = ""  

                            if actions2 and len(actions2) > 0:
                                action2 = actions2[0]
                            else:
                                action2 = "" 

                            combined_text = f"{subject1} {action1} and {connective} {action2}."
                            combined_sentences.append(combined_text)

                    for sentence in combined_sentences:
                        outfile.write(f'{sentence}\n')

                

    ########### create activity dataset ########
    save_root_path = '/root/autodl-tmp/StableMoFusion/activity'
    for split in ['train','val','test']:
        dataset = get_dataset(opt, split=split, mode='train', accelerator=accelerator)
        dataset.create_activity_dataset(save_root_path,split = split,num_transition_frames = 15, accelerator = accelerator)
        # process_txt_files(pjoin(save_root_path,split))
    ############################################
    # cap_path = '/root/autodl-tmp/StableMoFusion/activity/train/caption/'
    # # test_cap_path = '/root/autodl-tmp/StableMoFusion/scripts/fusion_caption/test_cap/'
    # fusion_prompt_path = '/root/autodl-tmp/StableMoFusion/scripts/fusion_caption/prompt/fusion_prompt.txt'
    # get_fusion_caption_from_cap(cap_path,fusion_prompt_path)
    ### run method
    # accelerate launch --config_file 1gpu.yaml --gpu_ids 0 -m scripts.create_dataset --name t2m_condunet1d --model-ema --dataset_name t2m
    ### debug method
    # python -m debugpy --listen 5678 --wait-for-client ./scripts/create_dataset.py
