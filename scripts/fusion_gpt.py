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
                            {"role": "user", "content": str(caps)}
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
                for ct in content_text:
                    f.write(f'{ct}')
        except Exception as e:
            print(f"Fusion caption failed: {e}")
                
cap_path = './activity/traincaption/'##改train，val，test
fusion_prompt_path = './fusion_prompt.txt'
get_fusion_caption_from_cap(cap_path,fusion_prompt_path)
