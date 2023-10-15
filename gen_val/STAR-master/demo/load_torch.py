# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman
from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch

# star = STAR(gender='female')
# betas = np.array([
#             np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
#                       2.20098416, 0.26102114, -3.07428093, 0.55708514,
#                       -3.94442258, -2.88552087])])
# num_betas=10
# batch_size=1
# m = STAR(gender='male',num_betas=num_betas)

# # Zero pose
# poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
# betas = torch.cuda.FloatTensor(betas)

# trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
# model = star.forward(poses, betas,trans)
# shaped = model.v_shaped[-1, :, :]

star = STAR(gender='female')
m = STAR(gender='female',num_betas=300)

path_m = r"E:\3D_HUMAN\Code\STAR-master\results\valid_subj_star_f.npz"  # contains beta and trans for each subj
path_pose_m = r"E:\3D_HUMAN\Code\train-SMPL\temp_files\pose_smpl_punch_kick.npy" # contains thetas for this sequence

subj_m = np.load(path_m)
betas = torch.cuda.FloatTensor(subj_m['betas'])
trans = torch.cuda.FloatTensor(subj_m['trans'])
pose_m = torch.cuda.FloatTensor(np.load(path_pose_m))

batch_size = len(pose_m)

# trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))

save_dict_m = dict()

name_lst_m = [
'50004',
'50020',
'50021',
'50022',
'50025',]

rotation_matrix = np.array([[1, 0, 0], 
                            [0, np.cos(np.pi/2), -np.sin(np.pi/2)], 
                            [0, np.sin(np.pi/2), np.cos(np.pi/2)]])

for i, beta in enumerate(betas):
    print(name_lst_m[i])
    beta_batch = beta.repeat(batch_size, 1)
    trans_batch = trans[i].repeat(batch_size, 1)
    # print(beta_batch.shape)
    result_v = star.forward(pose_m, beta_batch, trans_batch)
    result_np = result_v.detach().cpu().numpy()
    # result = model.v

    # AMASS's pose coefficient are rotated 90 degrees
    save_dict_m[name_lst_m[i]] = np.einsum('Nik,kj->Nij', result_np, rotation_matrix)
    # save_dict_m[name_lst_m[i]] = result_np

np.savez('./results/punch_kick_f_test.npz', **save_dict_m)
