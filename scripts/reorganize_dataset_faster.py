import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
## 并行多线程处理会出现数据错误和id错误
def copy_file(old_path, new_path):
    shutil.copy(old_path, new_path)

def reorganize_activity_dataset(activity_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, 'text')
    motion_dir = os.path.join(output_dir, 'motion')
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)

    train_ids, val_ids, test_ids = [], [], []

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(activity_dir, split)
        text_folder = os.path.join(split_path, 'text')
        motion_folder = os.path.join(split_path, 'motion')
        
        # 使用线程池
        with ThreadPoolExecutor() as executor:
            # 处理文本文件
            for filename in sorted(os.listdir(text_folder)):
                if filename.endswith('.txt'):
                    old_text_path = os.path.join(text_folder, filename)
                    new_file_id = len(os.listdir(text_dir)) + 1
                    new_text_name = f"{new_file_id:06d}.txt"
                    new_text_path = os.path.join(text_dir, new_text_name)

                    executor.submit(copy_file, old_text_path, new_text_path)

                    if split == 'train':
                        train_ids.append(new_text_name.split('.')[0])
                    elif split == 'val':
                        val_ids.append(new_text_name.split('.')[0])
                    elif split == 'test':
                        test_ids.append(new_text_name.split('.')[0])

            # 处理运动文件
            for filename in sorted(os.listdir(motion_folder)):
                if filename.endswith('.npy'):
                    old_motion_path = os.path.join(motion_folder, filename)
                    new_file_id = len(os.listdir(motion_dir)) + 1
                    new_motion_name = f"{new_file_id:06d}.npy"
                    new_motion_path = os.path.join(motion_dir, new_motion_name)

                    executor.submit(copy_file, old_motion_path, new_motion_path)

    # 保存train.txt, val.txt, test.txt文件
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_ids))

    # 复制Mean.npy和Std.npy文件
    shutil.copy(os.path.join(activity_dir, 'Mean.npy'), os.path.join(output_dir, 'Mean.npy'))
    shutil.copy(os.path.join(activity_dir, 'Std.npy'), os.path.join(output_dir, 'Std.npy'))

activity_directory = '/root/autodl-tmp/StableMoFusion/activity'  
output_directory = '/root/autodl-tmp/StableMoFusion/act' 
reorganize_activity_dataset(activity_directory, output_directory)
