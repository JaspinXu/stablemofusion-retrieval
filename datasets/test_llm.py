from openai import OpenAI
from torch import nn
import json
import os
import time

client = OpenAI(api_key=''
def get_local_text_from_text(text_path, out_local_text_path, prompt_path):
    try:
        with open(prompt_path, 'r') as f:
            prompt = f.read()
        os.makedirs(out_local_text_path, exist_ok=True)

        filename = 'test.txt'
        text_content = ['sitting down in place.',
'a person who is standing throws five punches in different directions with his left hand and follows that with two punches with his right hand.', 
'a person sitting down and trying to re-adjust the chair.',
'the person is doing a left hook and a left punch.']
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(text_content)}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
            )
        except Exception as e:
            print(f"Attempt failed: {e}")

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
        return True

    except Exception as e:
        print(f"Failed to generate local text from llm: {e}")
        return False

text_path = "/root/autodl-tmp/StableMoFusion/data/HumanML3D/texts/"
out_local_text_path = "/root/autodl-tmp/StableMoFusion/datasets/local_text_from_llm/"
prompt_path = "/root/autodl-tmp/StableMoFusion/scripts/fusion_caption/prompt/fusion_prompt.txt"

get_local_text_from_text(text_path, out_local_text_path, prompt_path)
# content = str(['sitting down in place.',
# 'a person who is standing throws five punches in different directions with his left hand and follows that with two punches with his right hand.', 
# 'a person sitting down and trying to re-adjust the chair.',
# 'the person is doing a left hook and a left punch.'])
# print(content,type(content))

# cap_file_path = '/root/autodl-tmp/StableMoFusion/datasets/024473.txt'
# content_text =[]
# with open(cap_file_path, 'r') as f:
#     # for line in f.readlines():
#     caps = []
#     for cap in f.readlines():
#         caps.append(cap)
#         if len(caps) == 2:
#     #         process_lines(lines)
#     #         caps = []
#     # caps = f.read()                      
#             print(str(caps))
#             content_text.append(str(caps))
#             caps = []
# print(content_text)
# with open(cap_file_path, 'w') as f:
#     for ct in content_text:
#         f.write(f'{str(ct)+'\n'}')
