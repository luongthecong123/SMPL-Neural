import torch
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch.optim import Adam, AdamW
from torch.nn.utils import prune
from torch.nn.functional import relu, leaky_relu
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import h5py

from models.neural_posedirs import Neural_posedirs
from utilities.init_full import get_dfaust_female
from utilities.data_loader import Subject_Regis
from utilities.utils import *
from config.config import *
from loss.criterion_neural import *
import matplotlib.pyplot as plt
from cycler import cycler

device = torch.device('cuda')
disable_posedirs = False

lambda_Y, lambda_J = cfg_multi_task["lambda_Y"], cfg_multi_task["lambda_J"]
lambda_P, lambda_W = cfg_multi_task["lambda_P"], cfg_multi_task["lambda_W"]
lambda_D = cfg_multi_task["lambda_D"]
lambda_edge = cfg_multi_task["lambda_edge"]
batch_size = cfg_hyper['batch_size']
epochs = cfg_hyper['epochs']

maleList = ['50002','50007','50009','50026','50027']
femaleList = ['50004','50020','50021','50022','50025']
allList = ['50002', '50007', '50009', '50026', '50027',
           '50004', '50020', '50021', '50022', '50025']

optimize_pose = False
optimize_trans = False

######################################### Prepare data #######################################################

# Train
path_train = r"./data/train_by_subj.h5py"
# file_train = h5py.File(path_train, 'r')
id_list_train, regis_arr_train, subset_list_train, first_regis_idx_dict_train = get_dfaust_female(path=path_train)

DFaust_data = Subject_Regis(id_list_train, regis_arr_train) # Pytorch dataset object

ids_dataloader = [DataLoader(
    dataset=Subset(DFaust_data, subject),
    batch_size=batch_size,
    shuffle=True
    )
    for subject in subset_list_train]

# Val
# All subjets have same pose (283,72), batch size equals the length of the pose
# Run 10 batches for 10 subjects. 
# trans term = 0 for everything since already match root joint of T_pose_blender with STAR inits.
path_val = "./data/valid_by_subj.h5py" # contains mocap registration
val_gt_mesh = h5py.File(path_val, 'r')
path_val_pose = "./data/pose_smpl_punch_kick.npy"  # contains pose for val (283, 72)
val_pose = torch.from_numpy(np.load(path_val_pose)).type(torch.float32).to(device)
file_val = h5py.File(path_val, 'r')
val_trans = torch.zeros((283,1,3)).type(torch.float32).to(device)

######################################### Load init data #######################################################
def Load_init():
        model_path='E:/3D_HUMAN/Code/SMPL-master/model_processed/model_f.pkl'
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

        device = torch.device('cuda')
        # T_hat & J_hat
        T_hat_dict = dict()
        J_hat_dict = dict()
        T_hat_data_np = np.load("./data/Blender_T_pose.npy")
        T_hat_data_np[:,1] = T_hat_data_np[:,1] - 1.15 # move it down a little bit

        # T_hat_data_np = np.load("./data/50004_jumping.npy")

        # T_hat_data_np[:,1] = T_hat_data_np[:,1] - 0.55  # Pre translate
        # T_hat_data_np[:,2] = T_hat_data_np[:,2] + 0.24

        # From SMPL
        # T_hat_data_np = params['v_template']
        # J_hat_data_np = np.matmul(params['J_regressor'].todense(),T_hat_data_np)
        # for key in femaleList:
        #     T_hat_dict[key] = torch.from_numpy(T_hat_data_np).type(torch.float32).to(device)
        #     J_hat_dict[key] = torch.from_numpy(J_hat_data_np).type(torch.float32).to(device)
        #     # J_hat_dict[key] = vert2joint_init(T_hat_blender).type(torch.float32).to(device)

        # # Ours 2 steps
        J_hat_dict_np = dict(np.load("./data/Final/J_hat_optimedOurs_Rigged_Blender_init_noposedirs.npz", allow_pickle=True))
        J_hat_dict_np = J_hat_dict_np["arr_0"].item()
        
        T_hat_dict_np = dict(np.load("./data/Final/T_hat_optimedOurs_Rigged_Blender_init_noposedirs.npz", allow_pickle=True))
        T_hat_dict_np = T_hat_dict_np["arr_0"].item()
        
        for key in femaleList:
            T_hat_dict[key] = torch.from_numpy(T_hat_dict_np[key]).type(torch.float32).to(device)
            J_hat_dict[key] = torch.from_numpy(J_hat_dict_np[key]).type(torch.float32).to(device)

        # Theta_big & trans_big
        path_theta_big = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\theta\theta_big_init_train_female.npy"
        theta_big = torch.from_numpy(np.load(path_theta_big)).type(torch.float32).to(device)

        path_trans_big = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\trans\trans_big_init_train_female.npy"
        trans_big = torch.from_numpy(np.load(path_trans_big)).type(torch.float32).to(device)
        trans_big = trans_big.unsqueeze(dim=1)

        # # Weights
        # path_weights = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\weights_geodesic_4joints.npy"
        # geodesic_init_weights = torch.from_numpy(np.load(path_weights)).type(torch.float32).to(device)
        

        # seg = json.load(open("./json/smpl_vert_segmentation.json"))
        # head_idx = seg["head"]
        # for idx in head_idx:
        #     geodesic_init_weights[idx] = 0.
        #     geodesic_init_weights[idx][15] = 1.

        # left_idx = seg["leftHandIndex1"]
        # for idx in left_idx:
        #     geodesic_init_weights[idx] = 0.
        #     geodesic_init_weights[idx][22] = 1.

        # right_idx = seg["rightHandIndex1"]
        # for idx in right_idx:
        #     geodesic_init_weights[idx] = 0.
        #     geodesic_init_weights[idx][23] = 1.
        # geodesic_init_weights[:,0] = 0.


        # # geodesic_init_weights = torch.from_numpy(params["weights"]).type(torch.float32).to(device)    
        # geodesic_init_weights = torch.load(r"E:\3D_HUMAN\Code\train-SMPL\optimized\optimized_weights_included\weights_max_0.pt"    )
        # geodesic_init_weights = geodesic_init_weights.to(device)

        # Ours
        path_ours = r"E:\3D_HUMAN\Code\train_SMPL_Final\data\Final\weights_optimedOurs_Rigged_Blender_init_noposedirs.pt"
        geodesic_init_weights = torch.load(path_ours).type(torch.float32).to(device)

        # # SMPL weights
        # geodesic_init_weights = torch.from_numpy(params['weights']).type(torch.float32).to(device)
        # Posedirs neural
        path_posedirs = r"E:\3D_HUMAN\Code\train_SMPL_Final\init\posedirs_neural_init_params_our_weights.pt"
        posedirs_statedict = torch.load(path_posedirs)
        return T_hat_dict, J_hat_dict, theta_big, trans_big, geodesic_init_weights, posedirs_statedict 

T_hat_dict, J_hat_dict, theta_big, trans_big, geodesic_init_weights, posedirs_statedict  = Load_init()
######################################### Optim variables #########################################

# T_hat & J_hat
for key in T_hat_dict.keys():
    T_hat_dict[key] = Variable(T_hat_dict[key], requires_grad=True)
    J_hat_dict[key] = Variable(J_hat_dict[key], requires_grad=True)
# Theta_big & trans_big
if optimize_pose == True:
    print("Optimize pose")
    theta_big_var = [Variable(theta_j, requires_grad=True) for theta_j in theta_big]  # a list
if optimize_pose == False:
    print("Doesn't optimize pose")
    theta_big_var = Variable(theta_big, requires_grad=False) # a tensor

if optimize_trans == True:
    print("Optimize trans")
    trans_big_var = [Variable(trans, requires_grad=True) for trans in trans_big] # a list
if optimize_trans == False:
    print("Doesn't optimize trans")
    trans_big_var = Variable(trans_big, requires_grad=False)
# Weights
weights_var = Variable(geodesic_init_weights, requires_grad=True)
# Posedirs neural
posedirs_var = Neural_posedirs()
posedirs_var.to(device)
posedirs_var.load_state_dict(posedirs_statedict)
prune.l1_unstructured(posedirs_var.layer_1, name='weight', amount=0.1)
posedirs_var.train()

val_trans_var = Variable(val_trans, requires_grad=True )
######################################### Optimizer ###############################################
# T_hat & J_hat
T_hat_optim_dict = dict()
J_hat_optim_dict = dict()
for key in T_hat_dict.keys():
    T_hat_optim_dict[key] = Adam([T_hat_dict[key]], lr=cfg_hyper['lr_T_hat'])
    J_hat_optim_dict[key] = Adam([J_hat_dict[key]], lr=cfg_hyper['lr_J_hat'])
# Theta_big & trans_big
if optimize_pose == True:
    theta_optim = Adam(theta_big_var, lr=cfg_hyper['lr_pose'])  # a list
if optimize_pose == False:
    theta_optim = Adam([theta_big_var], lr=cfg_hyper['lr_pose']) # a tensor

if optimize_trans == True:
    trans_optim = Adam(trans_big_var, lr=cfg_hyper['lr_trans']) # a list
if optimize_trans == False:
    trans_optim = Adam([trans_big_var], lr=cfg_hyper['lr_trans']) # a tensor
# Weights
weights_optim = Adam([weights_var], lr=cfg_hyper['lr_weights'])
weights_lr_scheduler = lr_scheduler.LinearLR(
    weights_optim, 
    start_factor=cfg_hyper['start_factor'], 
    end_factor=cfg_hyper['end_factor'], 
    total_iters=cfg_hyper['total_iters'])
# Posedirs neural
posedirs_optim = AdamW(posedirs_var.parameters(), lr=cfg_hyper['lr_posedirs'], weight_decay=1e-2)

val_trans_optim = Adam([val_trans_var], lr = 0.1)
######################################### Matplotlib dict ##############################################

loss_dict = {
    'data_loss': [],
    'symmetry_loss': [],
    'val_loss': []
}

acc_dict = {
    'acc_train': [],
    'acc_val': []
}

######################################### Train ########################################################

movie = torch.empty((0, 3))

flag = 0
for epoch in range(epochs):
    # torch.cuda.empty_cache ()
    iterators = list(map(iter, ids_dataloader)) 
    batch_num = 0
    d_loss_epoch = 0
    y_loss_epoch = 0
    acc_train = torch.empty((0, 6890)).to(device)

    ######### Train
    posedirs_var.train()
    while iterators:         
        iterator = np.random.choice(iterators)
        try:    
            regis, id, idx = next(iterator)
            id_0 = list(id)[0]
            T_hat_var = T_hat_dict[id_0]
            J_hat_var = J_hat_dict[id_0]
            
            if optimize_pose == True:
                pose_var = torch.index_select(torch.stack(theta_big_var), 0, idx.to(device))  # a list
            else: 
                pose_var = torch.index_select(theta_big_var, 0, idx.to(device)) # a tensor

            if optimize_trans == True:
                trans_var = torch.index_select(torch.stack(trans_big_var), 0, idx.to(device)) # a list
            else:
                trans_var = torch.index_select(trans_big_var, 0, idx.to(device))

            # print(pose_var.shape)
            # print(trans_var.shape)
            if flag == 0:
                with torch.no_grad():
                    loss_d, loss_y, euclid_dist = criterion(
                        T_hat=T_hat_var, 
                        J_hat=J_hat_var, 
                        weights=weights_var, 
                        posedirs_neural=posedirs_var, 
                        pose=pose_var, 
                        trans=trans_var, 
                        gt_regis=regis
                    )
            else:
                loss_d, loss_y, euclid_dist = criterion(
                        T_hat=T_hat_var, 
                        J_hat=J_hat_var, 
                        weights=weights_var, 
                        posedirs_neural=posedirs_var, 
                        pose=pose_var, 
                        trans=trans_var, 
                        gt_regis=regis
                    )    
            d_loss = lambda_D*loss_d.detach()
            y_loss = lambda_Y*loss_y.detach()

            print('id: {}, epoch: {}/{}, {}-th batch, loss_d: {:.3f}, loss_y: {:.3f}, weights lr: {}'
                  .format(id_0, epoch, epochs, batch_num, lambda_D*loss_d.item(), lambda_Y*loss_y.item(), weights_optim.param_groups[0]["lr"])) 
            
            def loss_closure():
                return lambda_D*loss_d + lambda_Y*loss_y
            
            d_loss_epoch += d_loss
            y_loss_epoch += y_loss

            batch_num += 1

            T_hat_optim_dict[id_0].zero_grad(set_to_none=True)
            J_hat_optim_dict[id_0].zero_grad(set_to_none=True)
            trans_optim.zero_grad(set_to_none=True)
            weights_optim.zero_grad(set_to_none=True) 
            posedirs_optim.zero_grad(set_to_none=True) 

            ########## Calulate graph
            if flag != 0:       
                loss_closure().backward()

                T_hat_optim_dict[id_0].step(loss_closure)
                # J_hat_optim_dict[id_0].step(loss_closure)
                # if optimize_trans:
                #     trans_optim.step(loss_closure)
                # if optimize_pose:
                #     posedirs_optim.step(loss_closure)
                # weights_optim.step(loss_closure)
                # posedirs_optim.step(loss_closure)
            with torch.no_grad():
                acc_train = torch.vstack((acc_train, euclid_dist))
                movie = torch.vstack((movie, T_hat_dict["50004"].detach().cpu()))


            # val_trans_optim.zero_grad(set_to_none=True)

            # del loss_d, loss_y, euclid_dist, d_loss, y_loss
            # torch.cuda.empty_cache ()

        except StopIteration:
            iterators.remove(iterator)
    weights_lr_scheduler.step()
    ########## Val
    posedirs_var.eval()
    val_loss = 0
    acc_val = torch.empty((0, 6890)).to(device)
    

    print("Start eval:")
    # with torch.no_grad():
    for subj in femaleList:
        print("Subj:", subj)
        val_gt_regis = torch.from_numpy(val_gt_mesh[subj][:]).to(device)
        # loss_d_val, euclid_dist_val = validation(
        loss_d_val, euclid_dist_val = validation(
                T_hat=T_hat_dict[subj], 
                J_hat=J_hat_dict[subj], 
                weights=weights_var, 
                posedirs_neural=posedirs_var, 
                pose=val_pose, 
                trans=val_trans_var, 
                gt_regis=val_gt_regis
            )

        val_trans_optim.zero_grad(set_to_none=True)

        loss_d_val.backward()
        val_trans_optim.step()

        with torch.no_grad():
            acc_val = torch.vstack((acc_val, euclid_dist_val))
            val_loss += loss_d_val*(64/283)

        # del loss_d_val, euclid_dist_val
        # torch.cuda.empty_cache ()
    ####### Print epoch results
    acc_train = torch.mean(acc_train)
    acc_val = torch.mean(acc_val)
    print('epoch: {}, data loss: {:.3f}, sym loss: {:.3f}, val loss: {:.3f}, train MABS: {:.3f}, val MABS: {:.3f} , num batches: {}'.format(
        epoch, d_loss_epoch/batch_num, y_loss_epoch/batch_num, val_loss/10, acc_train, acc_val, batch_num))
    
    ####### Save results to plt dict
    loss_dict['data_loss'].append(d_loss_epoch/batch_num)
    loss_dict['symmetry_loss'].append(y_loss_epoch/batch_num)
    loss_dict['val_loss'].append(val_loss/10)

    acc_dict['acc_train'].append(acc_train)
    acc_dict['acc_val'].append(acc_val)

    ### Change flag
    flag += 1

    

# torch.save(torch.stack(trans_big_var).detach().cpu(), './temp/trans_optimed.pt')

######################################### Plot results ##############################################

path = r"E:\3D_HUMAN\Code\train_SMPL_Final\optimized\SMPL"
T_pose_init_type = "SMPL_20epoch"

posedirs_var.eval()

monochrome = (cycler('color', ['k']) * cycler('marker', [' ', 'o']) *
              cycler('linestyle', ['-', ':', '--']))

for key in loss_dict.keys():
    try:
        loss_dict[key] = [loss.detach().cpu() for loss in loss_dict[key]]
    except:
        loss_dict[key] = [loss for loss in loss_dict[key]]
    plt.rc('axes', prop_cycle=monochrome)
    plt.plot(loss_dict[key], label=key)
    
plt.title('Loss theo epoch - ' + T_pose_init_type)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

for key in acc_dict.keys():
    try:
        acc_dict[key] = [1000*acc.detach().cpu() for acc in acc_dict[key]]
    except:
        acc_dict[key] = [1000*acc for acc in acc_dict[key]]
    plt.rc('axes', prop_cycle=monochrome)
    plt.plot(acc_dict[key], label=key)
    
plt.title('MABS theo epoch - ' + T_pose_init_type)
plt.xlabel('Epoch')
plt.ylabel('MABS (mm)')
plt.grid(True)
plt.legend()
plt.show()

######################################### Visualize results ##############################################

condition = weights_var < 0
count = torch.sum(condition)
print(f"Weights has {count} negative values")

# weights = leaky_relu(weights_var, 0.001)
# weights = weights / (torch.norm(weights, dim=1, keepdim=True, p = 1))    

weights = relu(weights_var)
weights = weights / (torch.norm(weights, dim=1, keepdim=True, p = 1) + 1e-8)

# weights = torch.abs(weights) / (torch.norm(weights, dim=1, keepdim=True, p = 1)) 
weights = weights_var
visualize_weights(weights.detach().cpu().numpy(), 9)


######################################### Save results ##############################################

# Save T_hat & J_hat
T_hat_dict_np = dict()
J_hat_dict_np = dict()
for key, value in T_hat_dict.items():
    T_hat_dict_np[key] = value.detach().cpu().numpy()
np.savez(path + '/T_hat_optimed'+ T_pose_init_type + '.npz', T_hat_dict_np)
for key, value in J_hat_dict.items():
    J_hat_dict_np[key] = value.detach().cpu().numpy()
np.savez(path + '/J_hat_optimed'+ T_pose_init_type + '.npz', J_hat_dict_np)
# Save weights
torch.save(weights.detach().cpu(), path + '/weights_optimed'+ T_pose_init_type + '.pt')
# Save posedirs state dict
prune.remove(posedirs_var.layer_1, name='weight')
torch.save(posedirs_var.state_dict(), path + '/posedirs_optimed'+ T_pose_init_type +'.pt')

########### Visualize train data

train_verts = []
train_joints = []
output_verts = []
gt_verts = []
pose_idx = []
id_list, regis_arr, subset_list, first_regis_idx_dict = get_dfaust_female(path=path_train)

for key in T_hat_dict:
    print("subject: ", key)
    train_verts.append(T_hat_dict[key])
    train_joints.append(J_hat_dict[key])
    # visualize(verts_joints.detach().cpu().numpy())
    gt = torch.from_numpy(regis_arr[first_regis_idx_dict[key]+230])
    gt_verts.append(gt)
    pose_idx.append(first_regis_idx_dict[key]+230)

if optimize_pose == False:
    pose_train = torch.index_select(theta_big_var, 0, torch.tensor(pose_idx).to(device))
else:
    pose_train = torch.index_select(torch.stack(theta_big_var), 0, torch.tensor(pose_idx).to(device))
if optimize_trans == False:
    trans_train = torch.index_select(trans_big_var, 0, torch.tensor(pose_idx).to(device))
else: 
    trans_train = torch.index_select(torch.stack(trans_big_var), 0, torch.tensor(pose_idx).to(device))
# trans_train = torch.index_select(trans_big_var, 0, torch.tensor(pose_idx).to(device))

# trans_train = torch.index_select(torch.stack(trans_big_var), 0, torch.tensor(pose_idx).to(device))
# pose_train = torch.index_select(theta_big_var, 0, torch.tensor(pose_idx).to(device))
gt_verts = torch.stack(gt_verts).to(device)
train_verts = torch.stack(train_verts)
train_joints = torch.stack(train_joints)

from models.smpl_pose_neural import SMPLModel
disable_posedirs = False
smpl_pose = SMPLModel(device=device,disable_posedirs=disable_posedirs)
smpl_pose.eval()

for i, train_vert in enumerate(train_verts):
    output, pose_offset = smpl_pose(
        T_hat=train_vert, 
        J_hat=train_joints[i], 
        weights=weights, 
        posedirs=posedirs_var, 
        pose=pose_train[i].unsqueeze(0), 
        trans=trans_train[i].unsqueeze(0))
    output_verts.append(output.squeeze(0))

output_verts = torch.stack(output_verts)    

train_verts = train_verts + trans_train

if disable_posedirs == True:
    print("Posedirs disabled for visualize test data")
    pose_blend_shapes = train_verts
else:
    print("Posedirs activate for visualize test data")
    pose_blend_shapes = train_verts + pose_offset.reshape(-1, 6890, 3)
total = torch.stack((train_verts, pose_blend_shapes, output_verts, gt_verts))

sequence_visualize_pro(total.detach().cpu())

########### Visualize val data

# train_verts = []
# train_joints = []
# output_verts = []
# gt_verts = []
# pose_idx = []

# for subj in femaleList:
#     print("Subj:", subj)
#     val_gt_regis = torch.from_numpy(val_gt_mesh[subj][:][50]).to(device)
#     output, pose_offset = smpl_pose(
#         T_hat=T_hat_dict[subj], 
#         J_hat=J_hat_dict[subj], 
#         weights=weights, 
#         posedirs=posedirs_var, 
#         pose=val_pose[50].unsqueeze(0), 
#         trans=val_trans[50].unsqueeze(0))
#     train_verts.append(T_hat_dict[subj])
#     gt_verts.append(val_gt_regis.squeeze(0))
#     output_verts.append(output.squeeze(0))

# train_verts = torch.stack(train_verts)
# gt_verts = torch.stack(gt_verts)
# output_verts = torch.stack(output_verts)

# if disable_posedirs == True:
#     pose_blend_shapes = train_verts
#     print("Posedirs disabled for visualize test data")
# else:
#     pose_blend_shapes = train_verts + pose_offset.reshape(-1, 6890, 3)
#     print("Posedirs activate for visualize test data")
# total = torch.stack((train_verts, pose_blend_shapes, output_verts, gt_verts))

# sequence_visualize_pro(total.detach().cpu(), distance=2.5)

# movie = movie.numpy()
# movie = movie.reshape(-1, 6890, 3)
# print("movie.shape: ", movie.shape)
# np.save("./temp/BlenderTo50004", movie)
# # animate_meshes(movie)