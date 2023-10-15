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
import numpy as np
import os
from convertors.losses import convert_smpl_2_star

########################################################################################################################
path_smpl_meshes = r'E:\3D_HUMAN\Code\train-SMPL\temp_files\valid_subj.npy'      #Path SMPL Meshes, a numpy array of SMPL verticies (n x 6890 x 3)

path_save_star_parms = r"E:\3D_HUMAN\Code\STAR-master\results\valid_subj_star.npz" #Path to save the star paramters

star_gender = 'female'   #STAR Model Gender (options: male,female,neutral).
MAX_ITER_EDGES = 100    #Number of LBFGS iterations for an on edges objective
MAX_ITER_VERTS = 1500   #Number of LBFGS iterations for an on vertices objective
NUM_BETAS = 10

# MAX_ITER_EDGES = 1   #Number of LBFGS iterations for an on edges objective
# MAX_ITER_VERTS = 1   #Number of LBFGS iterations for an on vertices objective
# NUM_BETAS = 10

if not os.path.exists(path_smpl_meshes):
    raise RuntimeError('Path to Meshes does not exist! %s'%(path_smpl_meshes))

opt_parms = {'MAX_ITER_EDGES':MAX_ITER_EDGES ,
             'MAX_ITER_VERTS':MAX_ITER_VERTS,
             'NUM_BETAS':NUM_BETAS,
             'GENDER':star_gender}
####################################### Find Beta for val data ###########################################
print('Loading the SMPL Meshes and ')
smpl = np.load(path_smpl_meshes)

smpl_m = smpl[:5]
smpl_f = smpl[5:]

opt_parms = {'MAX_ITER_EDGES':MAX_ITER_EDGES ,
             'MAX_ITER_VERTS':MAX_ITER_VERTS,
             'NUM_BETAS':NUM_BETAS,
             'GENDER':'male'}

path_save_star_parms_m = r"E:\3D_HUMAN\Code\STAR-master\results\valid_subj_star_m.npz"

np_poses , np_betas , np_trans , star_verts = convert_smpl_2_star(smpl_m,**opt_parms)
# results = {'poses':np_poses,'betas':np_betas,'trans':np_trans,'star_verts':star_verts}
# print('Saving the results %s.'%(path_save_star_parms))
# np.save(path_save_star_parms,results)

np.savez(path_save_star_parms_m, poses=np_poses, betas=np_betas, trans=np_trans, star_verts=star_verts)

opt_parms = {'MAX_ITER_EDGES':MAX_ITER_EDGES ,
             'MAX_ITER_VERTS':MAX_ITER_VERTS,
             'NUM_BETAS':NUM_BETAS,
             'GENDER':'female'}

path_save_star_parms_f = r"E:\3D_HUMAN\Code\STAR-master\results\valid_subj_star_f.npz"

np_poses , np_betas , np_trans , star_verts = convert_smpl_2_star(smpl_f,**opt_parms)

np.savez(path_save_star_parms_f, poses=np_poses, betas=np_betas, trans=np_trans, star_verts=star_verts)


