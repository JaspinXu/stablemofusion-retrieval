import numpy as np
import sys
import os
from os.path import join as pjoin

def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
    ## just calaulate train npy    
    # for file in file_list[:24546]:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)
    
    ## just calaulate train npy
    # np.save(pjoin(save_dir, 'train_mean.npy'), Mean)
    # np.save(pjoin(save_dir, 'train_std.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    ## calaulate all npy
    data_dir = '/root/autodl-tmp/StableMoFusion/act/motion/'
    save_dir = '/root/autodl-tmp/StableMoFusion/act/'
    mean, std = mean_variance(data_dir, save_dir, 22)
    ## just calaulate train npy
    # data_dir = '/root/autodl-tmp/StableMoFusion/data/HumanML3D/new_joint_vecs'
    # save_dir = '/root/autodl-tmp/StableMoFusion/data/HumanML3D/'
    # mean, std = mean_variance(data_dir, save_dir, 22)
#     print(mean)
#     print(Std)