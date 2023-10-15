batch_size_cfg = 64
cfg_multi_task = {
    'lambda_D': 15,
    'lambda_Y': 10, # joint symmetry term
    'lambda_J': 1000, # J_regressor*T - J
    'lambda_P': 5,  # regularize posedirs to zero   
    'lambda_W': 10,    # regularize weights to weights_init
    'lambda_edge': 1,
}
cfg_hyper = {
 'lr_pose' : 1e-4,
 'lr_T_hat' : 5e-3,
 'lr_J_hat' : 1e-2,
 'lr_trans' : 5e-3, # trans is already good
 'lr_posedirs': 1e-3, # learn slowly

 # learning rate decrease for weights
 'lr_weights': 0.001,
 'start_factor': 1.0,
 'end_factor': 0.01,
 'total_iters': 20, # decrease in n epochs

 'batch_size': batch_size_cfg,
 'epochs': 20
}

# batch_size_cfg = 64
# cfg_multi_task = {
#     'lambda_D': 15,
#     'lambda_Y': 10, # joint symmetry term
#     'lambda_J': 1000, # J_regressor*T - J
#     'lambda_P': 5,  # regularize posedirs to zero   
#     'lambda_W': 10,    # regularize weights to weights_init
#     'lambda_edge': 1,
# }
# cfg_hyper = {
#  'lr_pose' : 1e-4,
#  'lr_T_hat' : 1e-5,
#  'lr_J_hat' : 1e-3,
#  'lr_trans' : 5e-3, # trans is already good
#  'lr_posedirs': 1e-4, # learn slowly

#  # learning rate decrease for weights
#  'lr_weights': 0.001,
#  'start_factor': 0.1,
#  'end_factor': 0.001,
#  'total_iters': 10, # decrease in n epochs

#  'batch_size': batch_size_cfg,
#  'epochs': 35
# }