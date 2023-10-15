import torch
from loss.loss_func import *
from models.smpl_pose_neural import SMPLModel
from models.smpl_pose_SMPL import SMPLModel as SMPL
device = torch.device('cuda')
disable_posedirs = False
L1_loss = torch.nn.L1Loss(reduction="sum")
L2_loss = torch.nn.MSELoss(reduction="sum")
smooth_L1 = torch.nn.SmoothL1Loss(reduction="sum", beta = 20.0)

ours_model = SMPLModel(device=device, disable_posedirs = disable_posedirs)
smpl_pose = SMPL(device=device)
def cal_eu_dist(a_batch, b_batch):
    # Calculate euclidean distance between 2 batch of point clouds
    # Return a batchxnum_vertics tensor (batch_sizex6890)
    # Presumedly mean absolute vertex-to-vertex distance
    return torch.sqrt(torch.sum((a_batch-b_batch)**2,dim=2))

def criterion(T_hat, J_hat, weights, posedirs_neural, pose, trans, gt_regis):
    '''
    Only pose and gt_regis comes in as batch
    T_hat: 6890x3: vertices of T pose to optimize
    J_hat: 24x3: joints of T pose to optimize
    weights: 6890x24: weights - currently not to optimize
    posedirs_neural: a neural network model for P
    pose: batch_sizex72: pose to optimize
    trans: batch_sizex3: translation to optimize
    gt_regis: batchsizex6890x3: a batch of ground truth registration from data
    '''
    
    # output_regis, _ = ours_model(T_hat, J_hat, weights, posedirs_neural, pose, trans)
    output_regis, _ = smpl_pose(T_hat, pose, trans)

    loss_d = smooth_L1(output_regis, gt_regis)
    loss_y = Y_loss_pro(J_hat, T_hat)

    euclid_dist = cal_eu_dist(output_regis, gt_regis).detach()
    # euclid_dist = euclid_dist.detach().cpu()

    # del output_regis, _
    return loss_d, loss_y, euclid_dist

def validation(T_hat, J_hat, weights, posedirs_neural, pose, trans, gt_regis):
    '''
    Only pose and gt_regis comes in as batch
    T_hat: 6890x3: vertices of T pose to optimize
    J_hat: 24x3: joints of T pose to optimize
    weights: 6890x24: weights - currently not to optimize
    posedirs_neural: a neural network model for P
    pose: batch_sizex72: pose to optimize
    trans: batch_sizex3: translation to optimize
    gt_regis: batchsizex6890x3: a batch of ground truth registration from data
    '''
    
    output_regis, _ = ours_model(T_hat, J_hat, weights, posedirs_neural, pose, trans)
    
    loss_d = smooth_L1(output_regis, gt_regis)
    euclid_dist = cal_eu_dist(output_regis, gt_regis)
    # euclid_dist = euclid_dist.detach().cpu()
    # return euclid_dist
    # del output_regis, _
    return loss_d, euclid_dist


        