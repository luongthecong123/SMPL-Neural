import torch
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

from models.neural_posedirs import Neural_posedirs
from models.smpl_pose_LBS import SMPLModel as LBS
from models.smpl_pose_neural import SMPLModel as Posedirs_Neural
from models.smpl_pose_SMPL import SMPLModel as SMPL
from utilities.init_full import get_dfaust_female
from utilities.data_loader import Subject_Regis
from utilities.utils import *
import matplotlib.pyplot as plt
from cycler import cycler

device = torch.device('cuda')
femaleList = ['50004','50020','50021','50022','50025']
disable_posedirs = False
# Get Test data
batch_size = 64

path_test = r"E:\3D_HUMAN\Code\train_SMPL_Final\data\test_by_subj.h5py"
path_train = r"./data/train_by_subj.h5py"
id_list_train, regis_arr_train, subset_list_train, first_regis_idx_dict_train = get_dfaust_female(path=path_test)
print(regis_arr_train.shape)
DFaust_data = Subject_Regis(id_list_train, regis_arr_train) # Pytorch dataset object
ids_dataloader = [DataLoader(
    dataset=Subset(DFaust_data, subject),
    batch_size=batch_size,
    shuffle=True
    )
    for subject in subset_list_train]

path_pose = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\theta\theta_big_init_test_female.npy"
# path_pose = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\theta\theta_big_init_train_female.npy"
pose_np = np.load(path_pose)
pose_torch = torch.from_numpy(pose_np).type(torch.float32).to(device)
print("pose",pose_np.shape)
path_trans = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\trans\trans_big_init_test_female.npy"
# path_trans = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\trans\trans_big_init_train_female.npy"
trans_np = np.load(path_trans).reshape(-1,1,3)
trans_torch = torch.from_numpy(trans_np).type(torch.float32).to(device)
print("trans",trans_np.shape)
################################# Load Params ############################################

######### Ours
# T_hat, J_hat, weights step 2, posedirs neural
path_T_hat_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\T_hat_optimedOurs_BlenderToSubjects_Final.npz"
path_J_hat_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\J_hat_optimedOurs_BlenderToSubjects_Final.npz"
path_weights_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\weights_optimedOurs_BlenderToSubjects_Final.pt"
path_posedirs_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\posedirs_optimedOurs_BlenderToSubjects_Final.pt"

path_T_hat_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\T_hat_optimedOurs_SMPL_init.npz"
path_J_hat_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\J_hat_optimedOurs_SMPL_init.npz"
path_weights_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\weights_optimedOurs_SMPL_init.pt"
path_posedirs_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\posedirs_optimedOurs_SMPL_init.pt"

path_T_hat_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\rieng\T_hat_optimedOurs_separate_2Step_20epoch.npz"
path_J_hat_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\rieng\J_hat_optimedOurs_separate_2Step_20epoch.npz"
path_weights_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\rieng\weights_optimedOurs_separate_2Step_20epoch.pt"
path_posedirs_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\rieng\posedirs_optimedOurs_separate_2Step_20epoch.pt"

T_hat_dict_np_ours = np.load(path_T_hat_ours, allow_pickle=True)["arr_0"].item()
J_hat_dict_np_ours = np.load(path_J_hat_ours, allow_pickle=True)["arr_0"].item()

T_hat_dict_ours = dict()
J_hat_dict_ours = dict()
for key in femaleList:
    T_hat_dict_ours[key] = torch.from_numpy(T_hat_dict_np_ours[key]).type(torch.float32).to(device)
    J_hat_dict_ours[key] = torch.from_numpy(J_hat_dict_np_ours[key]).type(torch.float32).to(device)

# weights
weights_ours = torch.load(path_weights_ours).to(device)

# Posedirs_neural

posedirs_state_dict_ours = torch.load(path_posedirs_ours)
posedirs_ours = Neural_posedirs()
posedirs_ours.to(device)
posedirs_ours.load_state_dict(posedirs_state_dict_ours)
posedirs_ours.eval()

######### Maxplanc SMPL

# Only T_hat

path_T_hat_SMPL = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\SMPL\T_hat_optimedSMPL_20epoch.npz"
T_hat_dict_np_SMPL = np.load(path_T_hat_SMPL, allow_pickle=True)["arr_0"].item()
T_hat_dict_SMPL = dict()

for key in femaleList:
    T_hat_dict_SMPL[key] = torch.from_numpy(T_hat_dict_np_SMPL[key]).type(torch.float32).to(device)

######### Linear Blend Skinning

# T_hat, J_hat, weights step 1

path_T_hat_LBS = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\T_hat_optimedOurs_Rigged_Blender_init_noposedirs.npz"
path_J_hat_LBS = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\J_hat_optimedOurs_Rigged_Blender_init_noposedirs.npz"
path_weights_LBS = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\ours\chung\weights_optimedOurs_Rigged_Blender_init_noposedirs.pt"

T_hat_dict_np_LBS = np.load(path_T_hat_LBS, allow_pickle=True)["arr_0"].item()
J_hat_dict_np_LBS = np.load(path_J_hat_LBS, allow_pickle=True)["arr_0"].item()

T_hat_dict_LBS = dict()
J_hat_dict_LBS = dict()
for key in femaleList:
    T_hat_dict_LBS[key] = torch.from_numpy(T_hat_dict_np_LBS[key]).type(torch.float32).to(device)
    J_hat_dict_LBS[key] = torch.from_numpy(J_hat_dict_np_LBS[key]).type(torch.float32).to(device)

# weights
weights_LBS = torch.load(path_weights_LBS).to(device)

lbs = LBS(device=device)
smpl = SMPL(device=device)
ours = Posedirs_Neural(device=device, disable_posedirs = False)

import pickle
model_path='E:/3D_HUMAN/Code/SMPL-master/model_processed/model_f.pkl'
with open(model_path, 'rb') as f:
      params = pickle.load(f)
weights_SMPL = params["weights"]

# First regis idx:  {'50004': 0, '50020': 428, '50021': 718, '50022': 1055, '50025': 1457}
# id 50004 350; id 50021 1000; 50020 600 700; 50004 290
num = 700
subject = "50020"
print(f"subject {subject} num: {num},")
pose = pose_torch[num].unsqueeze(0)
trans = trans_torch[num].unsqueeze(0)
# trans = torch.zeros_like(trans)
gt_scan = torch.from_numpy(regis_arr_train[num]).unsqueeze(0).to(device)

print("First regis idx: ", first_regis_idx_dict_train)

def cal_eu_dist(a_batch, b_batch):
    # Calculate euclidean distance between 2 batch of point clouds
    # Return a batchxnum_vertics tensor (batch_sizex6890)
    # Presumedly mean absolute vertex-to-vertex distance
    return torch.sqrt(torch.sum((a_batch-b_batch)**2,dim=2))


# Get models output Ours
output_our, p_our = ours(
                        T_hat=T_hat_dict_ours[subject], 
                        J_hat=J_hat_dict_ours[subject], 
                        weights=weights_ours, 
                        posedirs=posedirs_ours, 
                        pose=pose, 
                        trans=trans)

print("Ours, MABS: ", torch.mean(cal_eu_dist(output_our, gt_scan)))

pose_offset = T_hat_dict_ours[subject] + p_our.reshape(-1, 6890, 3).squeeze(0)

# Stack to visualize ours
file_our = torch.vstack((T_hat_dict_ours[subject].unsqueeze(0) + trans, 
                    pose_offset.unsqueeze(0) + trans, 
                    output_our,
                    gt_scan
                    ))
print(file_our.shape)
sequence_visualize(file_our.detach().cpu())

# Get models output SMPL
output_smpl, p_smpl = smpl(
                        T_hat=T_hat_dict_SMPL[subject],
                        pose=pose, 
                        trans=trans)
print("SMPL, MABS: ", torch.mean(cal_eu_dist(output_smpl, gt_scan)))
# Stack to visualize smpl
pose_offset_smpl = T_hat_dict_SMPL[subject] + p_smpl.reshape(-1, 6890, 3).squeeze(0)
file_SMPL = torch.vstack((T_hat_dict_SMPL[subject].unsqueeze(0) + trans, 
                    pose_offset_smpl.unsqueeze(0) + trans, 
                    output_smpl,
                    gt_scan
                    ))
print(file_SMPL.shape)
sequence_visualize(file_SMPL.detach().cpu())

# Get models output LBS
output_lbs = lbs(
                        T_hat=T_hat_dict_LBS[subject],
                        J_hat=J_hat_dict_LBS[subject],
                        weights=torch.from_numpy(weights_SMPL).type(torch.float32).to(device),
                        pose=pose, 
                        trans=trans)
print("LBS, MABS: ", torch.mean(cal_eu_dist(output_lbs, gt_scan)))
# Stack to visualize lbs
file_lbs = torch.vstack((
  T_hat_dict_LBS[subject].unsqueeze(0) + trans,
  output_lbs,
  gt_scan  
))

sequence_visualize(file_lbs.detach().cpu())