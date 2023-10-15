import torch
import scipy.sparse as sp
import numpy as np
import pickle
import itertools
import json

src_path = "./model_processed/model_f.pkl"
with open(src_path, 'rb') as f:
    params = pickle.load(f)

faces = params["f"]

# J_regressor_init_path = "./data/J_regressor_init.npy"
# J_regressor_init = torch.from_numpy(np.load(J_regressor_init_path))


'''
get_vert_connectivity: This function takes the number of vertices and the face indices of a mesh, 
and returns a sparse matrix that indicates which vertices are connected by an edge.
'''
def get_vert_connectivity(num_verts, mesh_f):    
    vpv = sp.csc_matrix((num_verts,num_verts))
    def row(A):
        return A.reshape((1, -1))
    def col(A):
        return A.reshape((-1, 1))
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv

'''
get_verts_per_edge: This function takes the number of vertices and the face indices of a mesh, 
and returns an array of pairs of vertex indices that form an edge. It only returns edges where the 
first vertex index is smaller than the second one, to avoid duplicates.
'''
def get_verts_per_edge(num_verts,faces):
    vc = sp.coo_matrix(get_vert_connectivity(num_verts, faces))
    def row(A):
        return A.reshape((1, -1))
    def col(A):
        return A.reshape((-1, 1))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]]
    return result

'''
edge_loss: This function takes two meshes, d and smpl, and returns the difference between their edges. 
It uses the get_verts_per_edge function to get the edges for each mesh, and then subtracts them.
'''
def edge_loss_batch(d,smpl):
    vpe = get_verts_per_edge(6890, faces)
    edges_for = lambda x: x[:,vpe[:,0],:] - x[:,vpe[:,1],:]
    edge_obj = edges_for(d) - edges_for(smpl)
    return edge_obj


# non-batch version
def edge_loss(d,smpl):
    vpe = get_verts_per_edge(6890, faces)
    edges_for = lambda x: x[vpe[:,0],:] - x[vpe[:,1],:]
    edge_obj = edges_for(d) - edges_for(smpl)
    return edge_obj


'''
verts_loss: This function takes two meshes, d and smpl, and returns the sum of squared 
distances between their vertices.
'''
def verts_loss(d,smpl):
    return torch.sum((d-smpl)**2.0)

'''
v2v_loss: This function takes two meshes, d and smpl, 
and returns the mean of Euclidean distances between their vertices.
'''
def v2v_loss(d,smpl):
    return torch.mean(torch.sqrt(torch.sum((d-smpl)**2.0,axis=-1)))

def v2v_loss_sum(d,smpl):
    return torch.sum(torch.sqrt(torch.sum((d-smpl)**2.0,axis=-1)))

'''
Create a plane parallel to Oyz and goes through point p,
Doesn't need anymore because d always equal - joints[0][0]
'''
# def plane_parallel_to_oyz(p):
#   x = p[0]
#   a, b, c = torch.tensor([1, 0, 0]).type(torch.float32)
#   d = -x
#   print("d: ", d)
#   return a, b, c, d

'''
Takes the coefficients of a plane and points as input and 
returns the reflections of the points through the plane
'''
def householder_transform(points): 
    # n = torch.tensor([a, b, c]).to(points.device) # normal vector of the plane;
    # # print(f"n device {n.device}")
    # norm = torch.linalg.norm(n)
    # print("norm: ", norm)
    # if norm == 0:
    #     print("Warning: norm in householder_transform = 0")
    # n = n / norm
    # print("n: ", n)
    d_ = -points[0][0]
    # print("d = -points[0][0] = ", d_)
    n = torch.tensor([1.0, 0.0, 0.0], device=torch.device('cuda'))
    n_long = n.repeat(len(points), 1)

    dist = torch.einsum('ij, ij->i', n_long, points) + d_ # cross product of points with unit vector n
    n = n.unsqueeze(0) 
    dist = dist.unsqueeze(1) 
    reflection = points - 2 * torch.matmul(dist, n) 
    return reflection

'''
After reflection, the order of joints are mixed up (e.g. Left wrist and Right wrist), 
this function will correct it
'''
# def reorder(arr):
#     arr[[1, 2], :] = arr[[2, 1], :]
#     arr[[4, 5], :] = arr[[5, 4], :]
#     arr[[7, 8], :] = arr[[8, 7], :]
#     arr[[10, 11], :] = arr[[11, 10], :]
#     arr[[13, 14], :] = arr[[14, 13], :]
#     arr[[16, 17], :] = arr[[17, 16], :]
#     arr[[18, 19], :] = arr[[19, 18], :]
#     arr[[20, 21], :] = arr[[21, 20], :]
#     arr[[23, 23], :] = arr[[23, 22], :]
#     return arr

# def Y_loss(joints):
#     a, b, c, d = plane_parallel_to_oyz(joints[0])
#     reflections = householder_transform(a, b, c, d, joints)
#     reflections = reorder(reflections)
#     loss = verts_loss(joints, reflections)
#     return loss

###########################################################
device = torch.device('cuda')
joint_ord = json.load(open("./json/joint_order.json"))

leftJ = []
rightJ = []

for key in joint_ord.keys():
    if key[:4] == 'left':
        leftJ.append(joint_ord[key])
    if key[:4] == 'righ':
        rightJ.append(joint_ord[key])    

# leftJ contains indices for the joints on the left of the body
# with respect to all joints of shape 24,3
leftJ = torch.from_numpy(np.stack(leftJ)).squeeze().to(device)
rightJ = torch.from_numpy(np.stack(rightJ)).squeeze().to(device)
#-------------------------------------------------------------
vert_ord = json.load(open("./json/smpl_vert_segmentation.json"))
parts = ["Arm","Leg","ToeBase","Foot","Shoulder","HandIndex1","ForeArm","UpLeg","Hand"]

leftV = []
rightV = [] 

for part in parts:
    leftV.append(vert_ord["left" + part])
    rightV.append(vert_ord["right" + part])

leftV_lst = list(itertools.chain.from_iterable(leftV))
leftV_vctr = torch.tensor(leftV_lst, dtype=torch.int32).to(device)
# print(f"left V vctr shape: {leftV_vctr.shape}")
rightV_lst = list(itertools.chain.from_iterable(rightV))
rightV_vctr = torch.tensor(rightV_lst, dtype=torch.int32).to(device)
# print(f"right V vctr shape: {rightV_vctr.shape}")


def Y_loss_pro(joints, verts):


    ### Combine
    total = torch.vstack((joints, verts))
    reflections_total = householder_transform(total)
    reflections_J, reflections_V = torch.split(reflections_total,[24, 6890], dim=0)

    # reflections_J = householder_transform(a, b, c, d, joints)
    # print("householder")

    left_J =  torch.index_select(joints, 0, leftJ) - torch.index_select(reflections_J, 0, rightJ)
    right_J = torch.index_select(joints, 0, rightJ) - torch.index_select(reflections_J, 0, leftJ)
    left_right_J = torch.vstack((left_J, right_J))
    sym_joints = torch.sum((left_right_J)**2.0)
    # print("finished joints")


    # left_V =  torch.index_select(verts, 0, leftV_vctr) - torch.index_select(reflections_V, 0, rightV_vctr)
    # right_V = torch.index_select(verts, 0, rightV_vctr) - torch.index_select(reflections_V, 0, leftV_vctr)
    # left_right_V =  torch.vstack((left_V, right_V))
    # sym_verts = torch.sum((left_right_V)**2.0)

    # print("finished verts")



    sum = sym_joints
    # sum = lambda_U*sym_joints
    return sum
def J_loss(J_regressor_init, T_hat, J_hat):
    joints = torch.matmul(J_regressor_init, T_hat)
    J_loss = joints - J_hat
    return J_loss

if __name__ == "__main__":
    

    ############# sample code to test edge_loss ##########
    # from utils import *
    # import json
    # import pickle

    # joint_dict_explicit = json.load(open("./joint_dict_explicit.json"))
    # src_path = "./model_processed/model_f.pkl"
    # with open(src_path, 'rb') as f:
    #     params = pickle.load(f)

    # v_template = params["v_template"]
    # f = params["f"]
    # edge_pair = get_verts_per_edge(6890, f)
    # print(edge_pair.shape) # shape of (20664, 2)
    # print(edge_pair[0]) # [0 1]

    # def edge_loss(d,smpl):
    #     vpe = get_verts_per_edge(6890, faces)
    #     edges_for = lambda x: x[vpe[:,0],:] - x[vpe[:,1],:]
    #     edge_obj = edges_for(d) - edges_for(smpl)
    #     return edge_obj
    
    # d = np.load("./data/T_mu_pose.npy")

    # edge_obj = edge_loss(d,v_template)

    # print("edge_obj[0] :", edge_obj[0])
    # print("d[0] :", d[0]-d[1])
    # print("v_template[0] :", v_template[0]-v_template[1])    


    ############# sample code to test householder_transform ##########
    from utilities.utils import *
    '''
    this code runs in cpu
    '''
    import json
    import pickle
    
    device = torch.device('cuda')
    joint_dict_explicit = json.load(open("./json/joint_dict_explicit.json"))
    src_path = "./model_processed/model_f.pkl"
    with open(src_path, 'rb') as f:
        params = pickle.load(f)
    
    v_template = torch.from_numpy(params["v_template"]).type(torch.float32)
    print("v_template.shape: ", v_template.ndim)
    v_template = v_template.to(device=device)
    print("v_template.shape to device: ", v_template.ndim)
    joints = vert2joint_init(v_template, joint_dict_explicit, get_verts = False)

    joints = joints.to(device)
    sum_loss = Y_loss_pro(joints, v_template)
    print("1---------------------------------------")
    sum_loss = Y_loss_pro(joints, v_template)
    print("2---------------------------------------")
    sum_loss = Y_loss_pro(joints, v_template)
    print("3---------------------------------------")
    sum_loss = Y_loss_pro(joints, v_template)
    print("4---------------------------------------")
    print(sum_loss)

    # a, b, c, d = plane_parallel_to_oyz(joints[0])
    # reflections = householder_transform(a, b, c, d, joints)
    # # print(f"original {joints}")
    # # reflections_reord = reorder(reflections) # overwrite reflections
    # reflections_reord = np.copy(reflections)
    
    # loss = verts_loss(joints, reorder(reflections_reord))
    # # print(f"reordered: {reflections_reord}")
    # print(loss)

    # print("--------------------------------------------")

    # joint_ord = json.load(open("./json/joint_order.json"))

    # leftJ = []
    # rightJ = []

    # for key in joint_ord.keys():
    #     if key[:4] == 'left':
    #         leftJ.append(joint_ord[key])
    #     if key[:4] == 'righ':
    #         rightJ.append(joint_ord[key])    
    
    # leftJ = torch.from_numpy(np.stack(leftJ)).squeeze()
    # print("leftJ.dtype: ", leftJ.dtype)
    # rightJ = torch.from_numpy(np.stack(rightJ)).squeeze()

    
    # left =  torch.index_select(joints, 0, leftJ) - torch.index_select(reflections, 0, rightJ)
    # print("right joints")
    # print(torch.index_select(joints, 0, rightJ))
    # print("left joints reflected to the right")
    # print(torch.index_select(reflections, 0, leftJ))
    
    # print(f"left shape: {left.shape}")
    # right = torch.index_select(joints, 0, rightJ) - torch.index_select(reflections, 0, leftJ)
    # print(f"right shape: {right.shape}")
    # print(f"small: {torch.index_select(joints, 0, rightJ)[0][0]}")

    # twoside =  torch.vstack((left, right))
    # print(f"two_side: {twoside.shape}")

    # loss_2 = torch.sum((twoside)**2.0)

    # print(f"two side loss: {loss_2}")

    



    # # reflections.detach().numpy()
    # # joints.detach().numpy()
    # # plot_plane(a, b, c, d, joints)
    
    # print("--------------------------------------------")
    # vert_ord = json.load(open("./json/smpl_vert_segmentation.json"))
    # parts = ["Arm",
    # "Leg",
    # "ToeBase",
    # "Foot",
    # "Shoulder",
    # "HandIndex1",
    # "ForeArm",
    # "UpLeg",
    # "Hand"]


    # leftV = []
    # rightV = [] 
    # key_list = list(vert_ord.keys())
    # for part in parts:
    #     leftV.append(vert_ord["left" + part])
    #     rightV.append(vert_ord["right" + part])

    # # for key in vert_ord.keys():
    # #     if key[:4] == 'left':
    # #         print(key[4:])
    # #         leftV.append(vert_ord[key])
    # #     if key[:4] == 'righ':
    # #         # print(key[5:])
    # #         rightV.append(vert_ord[key])    
    # print("leftV")
    # for vert_seg in leftV:
    #     print(len(vert_seg))
    # print("rightV")
    # for vert_seg in rightV:
    #     print(len(vert_seg))    
    # # leftV = torch.from_numpy(np.stack(leftV)).squeeze()
    # # rightV = torch.from_numpy(np.stack(rightV)).squeeze()

    


    

    # print(len(leftV))
    # print(len(rightV)) 

    # leftV_lst = list(itertools.chain.from_iterable(leftV))
    # leftV_vctr = torch.tensor(leftV_lst, dtype=torch.int32)
    
    # print("leftV_vctr shape :", leftV_vctr.shape)  

    # rightV_lst = list(itertools.chain.from_iterable(rightV))
    # rightV_vctr = torch.tensor(rightV_lst, dtype=torch.int32)
    # print("rightV_vctr shape :", rightV_vctr.shape) 
    
    #########################################
    # reflections_V = householder_transform(a, b, c, d, v_template)
    # left =  torch.index_select(v_template, 0, leftV_vctr) - torch.index_select(reflections_V, 0, rightV_vctr)
    # print("right vert")
    # print(torch.index_select(joints, 0, rightJ)[:5])
    # print("left joints reflected to the right")
    # print(torch.index_select(reflections, 0, leftJ)[:5])
    
    # print(f"left shape: {left.shape}")
    # right = torch.index_select(v_template, 0, rightV_vctr) - torch.index_select(reflections_V, 0, leftV_vctr)
    # print(f"right shape: {right.shape}")
    # print(f"small: {torch.index_select(v_template, 0, rightV_vctr)[0][0]}")

    # twoside =  torch.vstack((left, right))
    # print(f"two_side: {twoside.shape}")

    # loss_2 = torch.sum((twoside)**2.0)

    # print(f"two side loss: {loss_2}")

    print("done")

