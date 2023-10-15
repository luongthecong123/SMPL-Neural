import torch
from utilities.utils import *
import h5py
import numpy as np
# from models.smpl_edge import SMPLModel_edge
# from models.smpl_pose_normalize import SMPLModel_norm
# from loss_func import *
from torch.autograd import Variable
import os
from tqdm import tqdm 
def get_dfaust(path=r"E:\3D_HUMAN\dataset\DFAUST\downsamp_data\DFAUST_by_subject_f_delta_80.hdf5"):
    '''
    Create an array of all the registration (regis) in the dataset and corresponding 
    subject id for that registration. Default data has float 32 precision
    Args:
        path: path to the hdf5 downsampled data, this data has keys as subject id, elements as registration
              created from function regis_by_subject in utils_data folder
    Returns:
        id_list: a list contains the subject id for each regis in regis_arr
        regis_arr: a numpy ndarray contains all the regis for all subject  
        subset_list: a list of list, its elements are lists of indices of the registration from the same subject
        first_regis_idx_dict: a dict: keys: id, values: index of first regis for id, the index of the first regis 
                         in file DFAUST_by_subject_f,index with respect to regis_arr
    # For example, if distance=2 in downsamp, first regis idx dict has the following value
    first_regis_idx_dict = {'50004': 0, '50020': 1880, '50021': 3160, '50022': 4460, '50025': 6399}
    '''
    id_list = []
    regis_arr = []

    with h5py.File(path, "r") as f:
        subject_list = list(f.keys())
        for subject in subject_list:
            # f[subject][:] returns the all the regis for that subject
            subject_regises = f[subject][:]
            for regis in subject_regises:   # regis has shape 6890,3
                id_list.append(subject)
                regis_arr.append(regis)

    regis_arr = np.stack(regis_arr, axis = 0)
    
    subset_list = []
    for subject in subject_list:
        subject_specific = []
        for i, id in enumerate(id_list):
            if id == subject:
                subject_specific.append(i)
        subset_list.append(subject_specific)  

    # the index of the first regis in file DFAUST_by_subject_f, index with respect to regis_arr
    first_regis_idx_dict = dict()
    with h5py.File(path, "r") as file:
        subject_list = list(file.keys())
        for subject in subject_list:
            # file[subject][:] returns the all the regis for that subject
            subject_regises = file[subject][:]
            for i, id in enumerate(id_list):
                if id == subject:
                    first_regis_idx_dict[subject] = i
                    break
           
    return id_list, regis_arr, subset_list, first_regis_idx_dict

def get_dfaust_female(path=r"E:\3D_HUMAN\dataset\DFAUST\downsamp_data\DFAUST_by_subject_f_delta_80.hdf5"):
    '''
    Create an array of all the registration (regis) in the dataset and corresponding 
    subject id for that registration. Default data has float 32 precision
    Args:
        path: path to the hdf5 downsampled data, this data has keys as subject id, elements as registration
              created from function regis_by_subject in utils_data folder
    Returns:
        id_list: a list contains the subject id for each regis in regis_arr
        regis_arr: a numpy ndarray contains all the regis for all subject  
        subset_list: a list of list, its elements are lists of indices of the registration from the same subject
        first_regis_idx_dict: a dict: keys: id, values: index of first regis for id, the index of the first regis 
                         in file DFAUST_by_subject_f,index with respect to regis_arr
    # For example, if distance=2 in downsamp, first regis idx dict has the following value
    first_regis_idx_dict = {'50004': 0, '50020': 1880, '50021': 3160, '50022': 4460, '50025': 6399}
    '''
    id_list = []
    regis_arr = []
    femaleList = [
                '50004',
                '50020',
                '50021',
                '50022',
                '50025',
                ]
    with h5py.File(path, "r") as f:
        subject_list = femaleList
        for subject in subject_list:
            # f[subject][:] returns the all the regis for that subject
            subject_regises = f[subject][:]
            for regis in subject_regises:   # regis has shape 6890,3
                id_list.append(subject)
                regis_arr.append(regis)

    regis_arr = np.stack(regis_arr, axis = 0)
    
    subset_list = []
    for subject in subject_list:
        subject_specific = []
        for i, id in enumerate(id_list):
            if id == subject:
                subject_specific.append(i)
        subset_list.append(subject_specific)  

    # the index of the first regis in file DFAUST_by_subject_f, index with respect to regis_arr
    first_regis_idx_dict = dict()
    with h5py.File(path, "r") as file:
        subject_list = femaleList
        for subject in subject_list:
            # file[subject][:] returns the all the regis for that subject
            subject_regises = file[subject][:]
            for i, id in enumerate(id_list):
                if id == subject:
                    first_regis_idx_dict[subject] = i
                    break
           
    return id_list, regis_arr, subset_list, first_regis_idx_dict


# def init_params(first_regis_idx_dict):
#     # weights
#     model_path='E:/3D_HUMAN/Code/SMPL-master/model_processed/model_f.pkl'
#     with open(model_path, 'rb') as f:
#         params = pickle.load(f)

#     # #Load ground truth weight
#     # # Create noisy weights
#     # geodesic_init_weights = add_noise(weights = params["weights"], noise = 1e-6)
#     # print(f"geodesic_init_weights weight: {geodesic_init_weights.shape}")

#     geodesic_init_weights = np.load('./init_data/weights_geodesic_4joints.npy')
#     seg = json.load(open("./json/smpl_vert_segmentation.json"))
#     head_idx = seg["head"]
#     for idx in head_idx:
#         geodesic_init_weights[idx] = 0.
#         geodesic_init_weights[idx][15] = 1.

#     left_idx = seg["leftHandIndex1"]
#     for idx in left_idx:
#         geodesic_init_weights[idx] = 0.
#         geodesic_init_weights[idx][22] = 1.

#     right_idx = seg["rightHandIndex1"]
#     for idx in right_idx:
#         geodesic_init_weights[idx] = 0.
#         geodesic_init_weights[idx][23] = 1.
#     geodesic_init_weights[:,0] = 0.
#     # geodesic_init_weights = params["weights"]
#     # print(f"geo desic weight: {geodesic_init_weights.shape}")

#     ############# J hat
#     joint_path_list = [r"E:\3D_HUMAN\dataset\DFAUST\temp\joints_50025.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\joints_50004.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\joints_50020.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\joints_50021.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\joints_50022.npy"]

#     init_J_hat_P = h5py.File("./init_data/J_hat_P.hdf5", "r")

#     joints_dict = dict()

#     v_template_maxplanc = params["v_template"]
#     joint_maxplanc = vert2joint_init(torch.from_numpy(v_template_maxplanc))

#     for path in joint_path_list:
#         # init pure LBS and add noise
#         # joints_dict[path[39:44]] = torch.from_numpy(np.load(path)).type(torch.float32).to(device)

#         # every subject use v template
#         joints_dict[path[39:44]] = joint_maxplanc.type(torch.float32).to(device)

#         # init with P and no noise - perfect
#         # joints_dict[path[39:44]] = torch.from_numpy(init_J_hat_P[path[39:44]][:]).type(torch.float32).to(device)

#     ############# T hat
#     v_template_path_list = [r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50022.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50025.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50004.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50020.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50021.npy"]

#     # init_T_hat_P = h5py.File("./init_data/T_hat_P.hdf5", "r")

#     v_template_dict = dict()


#     for path in v_template_path_list:
#         # init pure LBS and add noise
#         # v_template_dict[path[43:48]] = torch.from_numpy(np.load(path)).type(torch.float32).to(device)
        
#         # every subject use v template
#         v_template_dict[path[43:48]] = torch.from_numpy(v_template_maxplanc).type(torch.float32).to(device)

#         # init with P and no noise - perfect
#         # v_template_dict[path[43:48]] = torch.from_numpy(init_T_hat_P[path[43:48]][:]).type(torch.float32).to(device)

#     ############# theta_j init

#     # theta_path = r"E:\3D_HUMAN\Code\train-SMPL\init_data\theta_big_delta_80_3.pt"

#     theta_path = "./optimized/theta_pro.pt"
#     theta_big = torch.load(theta_path).to(device)

#     ############# trans
#     trans_path_list = [r"E:\3D_HUMAN\dataset\DFAUST\temp\trans_50022.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\trans_50025.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\trans_50004.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\trans_50020.npy",
#     r"E:\3D_HUMAN\dataset\DFAUST\temp\trans_50021.npy"]


#     init_trans_P = h5py.File("./init_data/trans_init_P.hdf5", "r")

#     trans_dict = dict()

#     for path in trans_path_list:
#         trans_dict[path[38:43]] = torch.from_numpy(np.load(path)).type(torch.float32).to(device)
#         # trans_dict[path[38:43]] = torch.zeros(3,).type(torch.float32).to(device)

#         # perfect trans:
#         # trans_dict[path[38:43]] = torch.from_numpy(init_trans_P[path[38:43]][:]).type(torch.float32).to(device)

#     trans_fdir = r"E:\3D_HUMAN\dataset\DFAUST\temp"

#     # Create a tensor that stores trans value for each pose, duplicates initialized one to match theta_big.shape[0]

#     first_idx_list = list(first_regis_idx_dict.keys())
#     # print("first_idx_list.device: ",first_idx_list[0].device)

#     # trans_big = torch.zeros((theta_big.shape[0], 1, 3)).type(torch.float32).to(device)
#     # for i, key in enumerate(first_idx_list):
#     #     start = first_regis_idx_dict[first_idx_list[i]]
#     #     try:
#     #         end = first_regis_idx_dict[first_idx_list[i+1]]
#     #     except:
#     #         end = trans_big.shape[0]
#     #     for j in range(start, end):
#     #             trans_big[j] = torch.from_numpy(np.load(trans_fdir + "/trans_" + key + ".npy")
#     #                                             ).type(torch.float32).to(device).unsqueeze(0)

#     trans_big = torch.load("./optimized/trans_pro.pt").to(device)
    
#     return geodesic_init_weights, v_template_dict, joints_dict, trans_big, theta_big




########################## Init Theta_j ##########################
########################## Init Theta_j ##########################
########################## Init Theta_j ##########################

# def init_theta_T_hat_trans_J_hat(regis_arr, id_list, first_regis_idx_dict):
#     '''
#     This function save init params to hard drive
#     Initializing params below:
#     1. theta big: pose for each registration
#     *** Finding pose for ground-truth registration, using edge loss ***
#         Optimize the theta (pose) so that the pose of the model's output (skinny woman) matches that
#         of the ground truth registration (fat woman). Input only pose to model

#         Returns:
#             theta_big: (n,72), type = torch.Tensor, require_grad = false 
#                         a tensor containing all the theta_j for n registrations 
        
#     2. T_hat: v_template for each subject
#     *** Find v_template for ground-truth registration, using vertice loss ***
#         Use only the first registration from of each subject
#         Already have the pose from step 1. Input v_template (skinny), pose, trans to model.
#         Loss = model output - ground truth
#         Now optimze the v_template & translation with vertices to make the skinny woman in T_pose fatter 
#         which in turn, will reduce the vertices loss.

#         Returns:
#             type = numpy.ndarray, shape (6890,3)
#             T_hat: a dict saved with h5py, each key is the subject id, key's value are v_template init for that subject
#             trans_init: a dict saved with h5py, each key is the subject id, key's value are translation for v_template
        
#     3. J_hat:         
#         Given T_hat, run vert2joint_init function to get the joint location.

#         Returns:
#             type = numpy.ndarray, shape (24,3)
#             J_hat: a dict saved with h5py, each key is the subject id, key's value are joints init for that subject

#     '''
#     theta_big = []
#     trans_init = dict()
#     T_hat = dict()
#     J_hat = dict()

#     device = torch.device('cuda')
#     # Create noisy T_template from Maxplanc V_template for T_hat init
#     model_path='E:/3D_HUMAN/Code/SMPL-master/model_processed/model_f.pkl'
#     with open(model_path, 'rb') as f:
#         params = pickle.load(f)
#     v_template = params["v_template"]
#     np.random.seed(42) ## Set seed form random
#     noise_array = np.random.normal(0, 1e-3, v_template.shape)
#     noisy_v_template = v_template + noise_array
#     noisy_v_template = torch.from_numpy(noisy_v_template).type(torch.float32).to(device)

#     smpl_edge = SMPLModel_edge(device = device)
#     smpl_norm = SMPLModel_norm(device = device)
    
#     for i, regis in enumerate(regis_arr):

#     ########################## Find pose that minimize edge loss ##############        
#         # id_list: a list contains the subject id for each regis in regis_arr

#         # This block of code returns the index of first regis for subject id
#         first_regis_idx = first_regis_idx_dict[id_list[i]] 
#         if i != first_regis_idx:    
#             continue       
#         # Load model

        

#         MAX_ITER_EDGES_ROOT = 10
#         MAX_ITER_EDGES_NORMAL = 30
#         learning_rate = 1e-1


#         global_pose = torch.cuda.FloatTensor(np.zeros(3,))
#         global_pose = Variable(global_pose, requires_grad=True)
#         joints_pose = torch.cuda.FloatTensor(np.zeros(69,))
#         joints_pose = Variable(joints_pose, requires_grad=True)

#         poses = torch.cat((global_pose, joints_pose), 0)
#         trans = torch.from_numpy(np.zeros(3)).type(torch.float32).to(device)

#         d = smpl_edge(poses, trans)

#         global_pose = torch.cuda.FloatTensor(np.zeros(3,))
#         global_pose = Variable(global_pose, requires_grad=True)
#         joints_pose = torch.cuda.FloatTensor(np.zeros(69,))
#         joints_pose = Variable(joints_pose, requires_grad=True)
#         poses = torch.cat((global_pose, joints_pose), 0)
#         d = smpl_edge(poses, trans)

#         optimizer = torch.optim.LBFGS([global_pose], lr=learning_rate)
#         # automatic gradient descent

#         ### Root joints
#         loss = torch.sum(edge_loss(d, regis) ** 2.0)
#         # print("root loss before: ", loss)
#         for t in range(MAX_ITER_EDGES_ROOT):
#             poses = torch.cat((global_pose, joints_pose), 0)
#             d = smpl_edge(poses, trans)

#             def edge_loss_closure():  # same as criterion
#                 loss = torch.sum(edge_loss(d, regis) ** 2.0)
#                 # print("loss: ", loss)
#                 return loss

#             optimizer.zero_grad() # avoid memory leak
#             edge_loss_closure().backward()
#             optimizer.step(edge_loss_closure)  # optimize on only root joint
#         loss = torch.sum(edge_loss(d, regis) ** 2.0)    
#         # print("root loss after - normal loss before: ", loss)

#         ##### Normal joints
#         optimizer = torch.optim.LBFGS([joints_pose], lr=learning_rate)
#         for t in range(MAX_ITER_EDGES_NORMAL):
#             poses = torch.cat((global_pose, joints_pose), 0)
#             d = smpl_edge(poses, trans)

#             def edge_loss_closure():
#                 loss = torch.sum(edge_loss(d, regis) ** 2.0)
#                 # print("loss: ", loss)
#                 return loss

#             optimizer.zero_grad()  # optimize on normal joints
#             edge_loss_closure().backward()
#             optimizer.step(edge_loss_closure)
#         loss = torch.sum(edge_loss(d, regis) ** 2.0)    
#         print(f"regis no. {i}, edge loss: {loss}")
#         theta_big.append(poses)

#     ########################## Find v_template that minimize vertice loss ##############
        
        
        
#         # Check if current iter is the first regis idx, if true, then execute the code
#         if i == first_regis_idx:
        
#             first_regis = regis_arr[first_regis_idx]
#             subject_0 = first_regis
            
#             rest_pose = noisy_v_template
#             rest_pose = Variable(rest_pose, requires_grad=True)
#             # Need optimize translation because root joint of v_template and regis are misaligned
#             # v_template we get from optimize and regis get from the scan

#             trans = Variable(trans, requires_grad=True)  
#             d = smpl_norm(poses, trans, rest_pose)

#             loss = torch.sum((d - subject_0) ** 2.0)
#             print("First loss: ", loss)

#             # Optimize rest pose vertices so that after forward, gives the same mesh as regis subject_0
#             optimizer = torch.optim.LBFGS([trans, rest_pose], lr=1e-1)

#             for t in tqdm(range(200)):
#                 d = smpl_norm(poses, trans, rest_pose)

#                 def verts_loss_closure():
#                     loss = torch.sum((d - subject_0) ** 2.0)
#                     # print("loss: ", loss)
#                     return loss

#                 optimizer.zero_grad()  # optimize on normal joints
#                 verts_loss_closure().backward()
#                 optimizer.step(verts_loss_closure)
            
#             loss = torch.sum((d - subject_0) ** 2.0)
#             print("v_template: vertice loss: ", loss)
#             T_hat[id_list[i]] = rest_pose

#             print("trans: ", trans)
#             trans_init[id_list[i]] = trans

#             print(f"Joint init for idx {i}")
#             joint_init = vert2joint_init(rest_pose)
#             J_hat[id_list[i]] = joint_init


#     # num = str(3)
#     num = "P"

#     # theta_big = torch.stack(theta_big).detach().cpu()
#     # torch.save(theta_big, "./init_data/theta_big_delta_80_"+ num + ".pt")

    

#     T_hat_tdir = "./init_data/T_hat_" + num + ".hdf5"
#     trans_tdir = "./init_data/trans_init_" + num + ".hdf5"
#     J_hat_tdir = "./init_data/J_hat_" + num + ".hdf5"

#     with h5py.File(T_hat_tdir, "w") as file:  # "w" is write
#         # Loop through the dictionary keys and values
#         for key, value in T_hat.items():
#             # Create a dataset with the same name as the key and the value as the data
#             file.create_dataset(key, data=value.detach().cpu().numpy())

#     with h5py.File(trans_tdir, "w") as file:  # "w" is write
#         # Loop through the dictionary keys and values
#         for key, value in trans_init.items():
#             # Create a dataset with the same name as the key and the value as the data
#             file.create_dataset(key, data=value.detach().cpu().numpy())

#     with h5py.File(J_hat_tdir, "w") as file:  # "w" is write
#             # Loop through the dictionary keys and values
#             for key, value in J_hat.items():
#                 # Create a dataset with the same name as the key and the value as the data
#                 file.create_dataset(key, data=value.detach().cpu().numpy())


# def init_theta(path):
#     '''
#     Save theta big and trans to folder for input h5py file
#     Args:
#         path: path to the h5py file that represents registration by subjects
#     Returns:

    
#     '''

if __name__ == "__main__":

    # # Code to test get_dfaust
    # _, _, _, first_regis_idx = get_dfaust()
    # print(first_regis_idx)
    # path=r"E:\3D_HUMAN\dataset\DFAUST\downsamp_data\DFAUST_by_subject_f.hdf5"
    # with h5py.File(path, "r") as f:
    #     subject_list = list(f.keys())
    #     sum = 0
    #     for subject in subject_list:
    #         # f[subject][:] returns the all the regis for that subject
    #         subject_regises = f[subject][:]
    #         print(len(subject_regises))
    #         sum += len(subject_regises)
    #         print(f"sum: {sum}")


    # # Code to test Theta init
    # id_list, regis_arr, subset_list, first_regis_idx_dict = get_dfaust()
    # for i, regis in enumerate(regis_arr):
    #     device = torch.device('cuda')
    #     # id_list: a list contains the subject id for each regis in regis_arr
    #     first_registration_idx = first_regis_idx_dict[id_list[i]]
    #     print(f"i: {i}, regis idx: {first_registration_idx}")
    #     first_registration = regis_arr[first_registration_idx]
        

    # print(first_registration.shape)
    device = torch.device('cuda')
    # id_list, regis_arr, subset_list, first_regis_idx_dict = get_dfaust()
    # regis_arr = torch.from_numpy(regis_arr).type(torch.float32).to(device)
    # init_theta_T_hat_trans_J_hat(regis_arr, id_list, first_regis_idx_dict)

    path = r"E:\3D_HUMAN\Code\train_SMPL_Final\data\train_by_subj.h5py"
    id_list, regis_arr, subset_list, first_regis_idx_dict = get_dfaust_female(path)
    print(regis_arr.shape)
    tdir = r"E:\3D_HUMAN\Code\train_SMPL_Final\temp\train_arr_female"
    np.save(tdir, regis_arr)


