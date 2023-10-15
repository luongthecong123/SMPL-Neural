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
pose_np = np.load(path_pose)
pose_torch = torch.from_numpy(pose_np).type(torch.float32).to(device)
print("pose",pose_np.shape)
path_trans = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\trans\trans_big_init_test_female.npy"
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

######################################### Trans Optimizer #######################################################
# optimize_pose = True
# if optimize_pose == True:
trans_big_var_ours = [Variable(trans, requires_grad=True) for trans in trans_torch] # a list
trans_big_var_SMPL = [Variable(trans, requires_grad=True) for trans in trans_torch] # a list
trans_big_var_LBS = [Variable(trans, requires_grad=True) for trans in trans_torch] # a list

optim_ours = Adam(trans_big_var_ours, lr=5e-5)
optim_SMPL = Adam(trans_big_var_SMPL, lr=5e-5)
optim_LBS = Adam(trans_big_var_LBS, lr=5e-5)


smooth_L1 = torch.nn.SmoothL1Loss(reduction="sum", beta = 20.0)

epochs = 10

lbs = LBS(device=device)
smpl = SMPL(device=device)
ours = Posedirs_Neural(device=device, disable_posedirs = False)

import pickle
model_path='E:/3D_HUMAN/Code/SMPL-master/model_processed/model_f.pkl'
with open(model_path, 'rb') as f:
      params = pickle.load(f)
weights_SMPL = torch.from_numpy(params["weights"]).type(torch.float32).to(device)
def cal_eu_dist(a_batch, b_batch):
    # Calculate euclidean distance between 2 batch of point clouds
    # Return a batchxnum_vertics tensor (batch_sizex6890)
    # Presumedly mean absolute vertex-to-vertex distance
    return torch.sqrt(torch.sum((a_batch-b_batch)**2,dim=2))

def test_params(T_hat_dict, J_hat_dict, model, optim, trans_torch_func, num_epoch):
    '''
    model: Choose between strings: "Ours" or "SMPL" or "LBS"
    optim: Optimizer object
    trans_torch_func: the variable for optimzation, a list of trans, 
    '''
    loss_test = []
    MABS_test = []
    for epoch in range(num_epoch):
        # print(" -------------This epoch starts------------------", epoch)
        batch_num = 0
        iterators = list(map(iter, ids_dataloader)) 
        loss_epoch = 0
        acc_test = torch.empty((0, 6890))
        while iterators:
            # print("While loop starts, batch num", batch_num)
            iterator = np.random.choice(iterators)
            try:    
                # print("Try starts")
                regis, id, idx = next(iterator)
                id_0 = list(id)[0]
                T_hat_var = T_hat_dict[id_0]
                J_hat_var = J_hat_dict[id_0]
                pose_var = torch.index_select(pose_torch, 0, idx.to(device))
                trans_var = torch.index_select(torch.stack(trans_torch_func), 0, idx.to(device))

                # Ours
                if model == "Ours":
                    outputs, _ = ours(
                        T_hat=T_hat_var, 
                        J_hat=J_hat_var, 
                        weights=weights_ours, 
                        posedirs=posedirs_ours, 
                        pose=pose_var, 
                        trans=trans_var)
                # SMPL
                elif model == "SMPL":
                    outputs, _ = smpl(
                        T_hat=T_hat_var,
                        pose=pose_var, 
                        trans=trans_var)
                else:
                    outputs = lbs(
                        T_hat=T_hat_var,
                        J_hat=J_hat_var,
                        weights=weights_SMPL,
                        pose=pose_var, 
                        trans=trans_var)

                optim.zero_grad(set_to_none=True)   
                loss = 15*smooth_L1(outputs, regis)
                loss.backward()
                optim.step()
                euclid_dist = cal_eu_dist(outputs, regis).detach().cpu()
                loss_epoch += loss.detach().cpu()
                
                
                batch_num += 1

                with torch.no_grad():
                    acc_test = torch.vstack((acc_test, euclid_dist))
                print("Model: {}, epoch {}/{}, {}-th batch, subject {}, loss: {:.3f}"
                        .format(model, epoch, num_epoch, batch_num, id_0, loss.item()))
            except Exception as e:
                print("----------Exception: ", e)
                print("except starts")
                iterators.remove(iterator)
        # print("While ends")    
        acc_test = torch.mean(acc_test)
        print("Model: {}, epoch: {}, test loss: {:.3f}, test MABS {:.6f}, num_batches {}"
                .format(model, epoch, loss_epoch/batch_num, acc_test, batch_num))
        loss_test.append(loss_epoch/batch_num)
        MABS_test.append(acc_test*1000)
        # print("For loop ends")
    return loss_test, MABS_test

loss_ours, MABS_ours  = test_params(T_hat_dict_ours, J_hat_dict_ours, "Ours", optim_ours, trans_big_var_ours, 2)

# # J_hat_dict_ours for SMPL is just for syntax

print(" ------------------Start          SMPL-----------------------")
loss_smpl, MABS_smpl  = test_params(T_hat_dict_SMPL, J_hat_dict_ours, "SMPL", optim_SMPL, trans_big_var_SMPL, 2)
print(" ------------------Start          LBS-----------------------")
loss_lbs, MABS_lbs  =   test_params(T_hat_dict_LBS, J_hat_dict_LBS, "LBS", optim_LBS, trans_big_var_LBS, 2)
                    
# monochrome = (cycler('color', ['k']) * cycler('marker', [' ', 'o']) *
#               cycler('linestyle', ['-', ':', '--']))    

# plot loss 
# plt.rc('axes', prop_cycle=monochrome)            
plt.plot(loss_ours, label="loss Ours")
# plt.rc('axes', prop_cycle=monochrome)  
plt.plot(loss_smpl, label="loss SMPL")
# plt.rc('axes', prop_cycle=monochrome)  
plt.plot(loss_lbs, label="loss LBS")

plt.title('Testing - Loss theo epoch ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# plot MABS

# plt.rc('axes', prop_cycle=monochrome)            
plt.plot(MABS_ours, label="MABS Ours")
# plt.rc('axes', prop_cycle=monochrome)  
plt.plot(MABS_smpl, label="MABS SMPL")
# plt.rc('axes', prop_cycle=monochrome)  
plt.plot(MABS_lbs, label="MABS LBS")

plt.title('Testing - MABS theo epoch ')
plt.xlabel('Epoch')
plt.ylabel('MABS (mm)')
plt.grid(True)
plt.legend()
plt.show()