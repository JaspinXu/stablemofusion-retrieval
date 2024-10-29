from utils.motion_process import recover_from_ric
import numpy as np
import torch
from utils.plot_script import draw_to_batch
import time 

# def motion_linear_interpolate_torch(motion1, motion2, num_transition_frames):
#     last_frame_motion1 = motion1[-1]
#     first_frame_motion2 = motion2[0]
#     start1 = time.time()
#     # transition_frames = np.linspace(last_frame_motion1.cpu().numpy(), first_frame_motion2.cpu().numpy(), num=num_transition_frames)
#     transition_frames = torch.empty((num_transition_frames,0)).cuda()
#     for i in torch.arange(len(last_frame_motion1)).cuda():
#         transition_frames_col = torch.linspace(last_frame_motion1[i],first_frame_motion2[i],steps=num_transition_frames).cuda()
#         transition_frames = torch.cat((transition_frames, transition_frames_col.unsqueeze(1)), dim=1).cuda()
#     concatenated_motion = torch.cat((motion1, transition_frames, motion2), dim=0).cuda()
#     # concatenated_motion = torch.cat((motion1, torch.from_numpy(transition_frames).float().cuda(), motion2), dim=0).cuda()
#     end1 = time.time()
#     print(end1 - start1)
#     return concatenated_motion

# def motion_linear_interpolate_numpy(motion1, motion2, num_transition_frames):
#     last_frame_motion1 = motion1[-1]
#     first_frame_motion2 = motion2[0]
#     start2 = time.time()
#     transition_frames = np.linspace(last_frame_motion1.cpu().numpy(), first_frame_motion2.cpu().numpy(), num=num_transition_frames)
#     # transition_frames = torch.empty((num_transition_frames,0))
#     # for i in range(len(last_frame_motion1)):
#     #     transition_frames_col = torch.linspace(last_frame_motion1[i],first_frame_motion2[i],steps=num_transition_frames).cuda()
#     #     transition_frames = torch.cat((transition_frames, transition_frames_col.unsqueeze(1)), dim=1).cuda()
#     # concatenated_motion = torch.cat((motion1, transition_frames, motion2), dim=0).cuda()
#     concatenated_motion = torch.cat((motion1, torch.from_numpy(transition_frames).float().cuda(), motion2), dim=0).cuda()
#     end2 = time.time()
#     print(end2 - start2)
#     return concatenated_motion

# def motion_fusion_smooth_interpolate(motion_list, num_transition_frames):
#     smooth_motion = motion_list[0]
    
#     for i in range(1, len(motion_list)):
#         smooth_motion = motion_linear_interpolate(smooth_motion, motion_list[i], num_transition_frames)
    
#     return smooth_motion
# frames_length = 588
motion= torch.from_numpy(np.load('/root/autodl-tmp/StableMoFusion/act/motion_re/000012.npy')).float().cuda()
pred_xyz = recover_from_ric(motion, 22).cuda()
xyz = pred_xyz.reshape(1, -1, 22, 3).cuda()
draw_to_batch(xyz.detach().cpu().numpy(),'a', [f'motionre000012.gif'])

# if len(cat_motion) >= frames_length:
#     # idx = random.randint(0, motion_length - frames_length)
#     idx = torch.randint(0, len(cat_motion) - frames_length, (1,), device='cuda').item()
#     cat_motion = cat_motion[idx: idx + frames_length]
# else:
#     cat_motion = torch.cat((cat_motion,
#                             torch.zeros((frames_length - len(cat_motion), cat_motion.shape[1])).cuda()), dim=0).cuda()
# assert len(cat_motion) == frames_length

# pred_xyz = recover_from_ric(cat_motion, 22)
# xyz = pred_xyz.reshape(1, -1, 22, 3)

# pose_vis = draw_to_batch(xyz.detach().cpu().numpy(),'a', ['example.gif'])