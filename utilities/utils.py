import torch
import numpy as np
import json

# joint_dict_explicit = json.load(open("./json/joint_dict_explicit.json"))


###############################################################################
############################ INITIALIZATION FUNCTIONS #########################
###############################################################################

def vert2joint_init(vertices, ring_ids = json.load(open("./json/joint_dict_explicit.json")), get_verts = False):
    '''
    Args:   
        vertices: body vertices, shape (6890,3)
        ring_ids: a dict containing the common vertices between body parts (ring shape)
        get_verts: if True, will return a tuple (verts_total, joints_total)
        device: cuda
    Returns: 
        joints_total: joint initials, computed by getting average of the ring shapes, shape (24, 3)
        if get_verts = True, will return a tuple (verts_total, joints_total)
    '''
    # use torch.tensor to convert numpy arrays to pytorch tensors
    joints_total = torch.tensor([], dtype = torch.float32).reshape(0, 3).to(vertices.device)
    verts_total = torch.tensor([], dtype = torch.float32).reshape(0, 3).to(vertices.device)
    

    for key in ring_ids:
        # use torch.cat to concatenate tensors along a given dimension
        verts = torch.cat([vertices[ring_ids[key]]], dim=0)
        joints = torch.mean(verts, dim = 0).reshape((1,3))
        verts_total = torch.cat((verts_total, verts), dim=0)  
        joints_total = torch.cat((joints_total, joints), dim=0)

    # root is average of 3 joints number 1, 2, 3. In joints_total, their index are: 0, 1, 2
    root_total = joints_total[0:3]
    root_joint = torch.mean(root_total, dim = 0).reshape((1,3))
    joints_full = torch.cat((root_joint, joints_total), dim=0)
    if get_verts:
        return (verts_total, joints_full)
    return joints_full


def get_mean_mesh_joint(data_dir):  #data_dir = "./data/"
    '''
    Function to calculate the mean mesh from all subject's registration
    '''
    import os
    import json
    joint_dict_explicit = json.load(open("./json/joint_dict_explicit.json"))
    # define number of desired registrations
    subjects_list = os.listdir(data_dir + "sample_data/")

    T_mu_pose = []
    sum = 0
    for subject in subjects_list:
        subject_path = os.path.join(data_dir, "sample_data", subject)    

        # use torch.load to load pytorch tensors from files
        subject_arr = torch.from_numpy(np.load(subject_path))
        subject_len = len(subject_arr)

        T_mu_pose_subject = torch.mean(subject_arr, dim = 0)
        T_mu_pose_subject_weighted = T_mu_pose_subject*subject_len
        T_mu_pose.append(T_mu_pose_subject_weighted)
        sum += subject_len
    # use torch.stack to stack a sequence of tensors along a new dimension
    T_mu_pose = torch.stack(T_mu_pose)          
    T_mu_pose = torch.mean(T_mu_pose, dim = 0)/sum
    J_mu_pose = vert2joint_init(T_mu_pose, joint_dict_explicit)

    return (T_mu_pose, J_mu_pose)

def get_J_regressor_init(mesh, joint_init):
    # J_regressor, residuals, rank, s = np.linalg.lstsq(mesh.T, joint_init.T, rcond=None)
    J_regressor, residuals, rank, s = torch.linalg.lstsq(mesh.T, joint_init.T)
    J_regressor = J_regressor.T
    return J_regressor


def add_noise(weights, noise, seed=42):  
    '''
    Add noise to non-zero elements in weights using Gaussian distribution
    Args:
        weights: np array, weights from SMPL params
        noise: float, the standard deviation of the distribution, noise = 1e-6 or lower
        seed: int, set the seed for numpy random function
    Returns:
        weights: np array, new weights
    Maxplanc weights statistic:
    max(weights):  0.9999537426142646
    min(weights) non-zero :  3.7671318480327008e-06
    '''
    # create a boolean mask of non-zero elements
    mask = weights != 0
    np.random.seed(seed)
    # create a random noise array of the same shape as weights
    noise_array = np.random.normal(0, noise, weights.shape)
    '''
    The probability density function - Gaussian distribution: 
    0: mean ("centre") of the distribution
    noise: standard deviation (spread of "width") of the distribution
    weights.shape: output shape of array, if = None, output a single value
    '''
    # add the noise only to the non-zero elements using the mask
    weights[mask] += noise_array[mask]
    return np.abs(weights)

def init_weight_geodesic():
    # Load the ring
    import pygeodesic
    import pygeodesic.geodesic as geodesic
    

    params = get_params_joint_dict()
    import json
    with open("./json/joint_dict_explicit.json") as f:
        joint_dict_explicit = json.load(f)    

    v_template = params["v_template"]
    faces = params["f"]
    joints_init = vert2joint_init(torch.from_numpy(v_template))
    geoalg = geodesic.PyGeodesicAlgorithmExact(v_template, faces)


    weights_init = [] # transpose later
    joints_init = joints_init.numpy()

    for i, joint in enumerate(joints_init):
        print(f"joint number {0}, shape {joint.shape}")
        print(i)
        if i==0:
            print("first")
            ring = joint_dict_explicit["1"] + joint_dict_explicit["2"] + joint_dict_explicit["3"]
            print(ring)
        else:
            print("the rest")
            ring = joint_dict_explicit[str(i)]
            print(ring)
        min_arr = []
        j2ring_distance = []

        for vertex_idx in ring:
            # For each vertex in the ring, find its geodesic distance to all other vertices
            # Save all of them to an array to find the min distance later
            v2all_distance, _ = geoalg.geodesicDistances([vertex_idx]) # shape 6890,
            min_arr.append(v2all_distance)

            # distance of joint to the current vertex in ring
            j2ring = np.sqrt(np.sum((joint - v_template[vertex_idx])**2))
            # print(f"joint to ring vertex distance: {j2ring}")
            j2ring_distance.append(j2ring)
        min_arr = np.vstack(min_arr) # min_arr shape: n, 6890, then find the min geodesic distance along the column
        min_arr = np.min(min_arr, axis=0, keepdims=True)  # shape 6890,
            
        min_arr += min(j2ring_distance)  # Add up to return the min geodesic distance from the joint to the ring to other

        # normalize all distance to smallest distance after add
        min_arr /= np.min(min_arr)
        weights_init.append(1/min_arr)

    weights_init = np.vstack(weights_init)
    weights_init_og = weights_init.transpose()

    weights_init_og_percentile = []
    for row in weights_init_og:  # 6890, 24
        # print("row ", row)
        ind = np.argpartition(row, -2)[-2:]
        row[~np.isin(np.arange(len(row)), ind)] = 0
        # print(row)
        weights_init_og_percentile.append(row)
        # print("new_row ", row)
    weights_init_og_percentile = (np.vstack(weights_init_og_percentile))

    return weights_init_og_percentile  


       


###############################################################################
############################## GET PARAMS FUNCTIONS ###########################
###############################################################################

def get_sample_mesh(path="./data/sample_data"):  # path = "./data/sample_data"
    import os
    import numpy as np
    '''
    Returns a mesh 6890x3 from a dataset to test code 
    '''

    # 0.npy -> 223, 6890,3

    subject_path_list = os.listdir(path)
    subject_0_path = subject_path_list[0]
    regist_0 = np.load(os.path.join(path, subject_0_path))[0]
    return regist_0

def get_params_joint_dict(joint_dict = False): 
    '''
    get SMPL params and joint dict explicit (this used for getting joints from segment's ring)
    Args: joint_dict: If True, returns the joint_dict_explicit
    Returns: params: SMPL params
             joint_dict: joint_dict_explicit, used for getting joints from segment's ring

    '''
    import json
    import pickle
    import numpy as np

    
    src_path = r"E:\3D_HUMAN\Code\train-SMPL\model_processed\model_f.pkl"
    with open(src_path, 'rb') as f:
        params = pickle.load(f)
    if joint_dict:
        joint_dict_explicit = json.load(open(r"E:\3D_HUMAN\Code\train-SMPL\json\joint_dict_explicit.json"))
        return params, joint_dict_explicit
    else:
        return params
    
###############################################################################
############################# VISUALIZATION FUNCTIONS #########################
###############################################################################

def animate_meshes_o3d(verts, time_sleep=False):
    import open3d as o3d
    import time
    '''
    args: verts, shape nx6890x3, sequence to animate
    
    '''
    params = get_params_joint_dict()
    faces = params["f"]
    vis = o3d.visualization.Visualizer()
    vis.create_window(height = 900, width = 720)

    pcd = o3d.geometry.TriangleMesh()
    # pcd = o3d.visualization.Material()

    material = o3d.visualization.Material()
    # material.base_color = [1.0, 0.0, 0.0]
    # material.specular_color = [1.0, 1.0, 1.0]
    # material.shininess = 50

    # Assigner le matériau au maillage
    # pcd.material = material

    vis.add_geometry(pcd)

    for i in range(len(verts)):
        pcd.vertices = o3d.utility.Vector3dVector(verts[i])
        pcd.triangles = o3d.utility.Vector3iVector(faces)

        if i == 0:
            vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if time_sleep:
            time.sleep(time_sleep)

# Define a function to animate meshes with trimesh
# def animate_meshes(point_clouds):
#     import numpy as np
#     import trimesh
#     import trimesh.transformations as tf


#     params = get_params_joint_dict()
#     faces = params["f"]

#     # Create a list of meshes from the point clouds and faces
#     meshes = []
#     for point_cloud in point_clouds:
#         # Create a mesh object from the point cloud and faces
#         mesh = trimesh.Trimesh(point_cloud, faces)
        
#         # Add the mesh to the list of meshes
#         meshes.append(mesh)
    
#     # Create a scene object
#     scene = trimesh.Scene()
    
#     # Add the first mesh to the scene
#     scene.add_geometry(meshes[0])
    
#     # Set the camera position and orientation
#     scene.set_camera(distance=4)
#     # Define a function to update the scene with each frame
#     def update_scene(scene):
#         # Get the current frame index
#         i = scene.frame_index
        
#         # Remove the previous frame from the scene
#         scene.delete_geometry(next(iter(scene.geometry.keys())))
        
#         # Add the next frame to the scene
#         scene.add_geometry(meshes[i])
        
#         # Increment the frame index modulo n
#         scene.frame_index = (i + 1) % len(meshes)
    
#     # Set the update function for the scene
#     scene.update_function = update_scene
    
#     # Set the initial frame index for the scene
#     scene.frame_index = 0
    
#     # Show the scene in a 3D viewer
#     scene.show(viewer='gl', smooth=False)

def animate_point_clouds(verts, time_sleep=0.1):
    import open3d as o3d
    import time
    '''
    args: verts, shape nx6890x3, sequence to animate
    
    '''
    params = get_params_joint_dict()
    faces = params["f"]
    vis = o3d.visualization.Visualizer()
    vis.create_window(height = 900, width = 720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    for i in range(len(verts)):
        pcd.points = o3d.utility.Vector3dVector(verts[i])
        if i == 0:
            vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if time_sleep:
            time.sleep(time_sleep)


def visualize(vertices, window_name = "", height = 900, width = 720):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    '''
    Args: 
        vertices: np array matrix with shape [Nx3]
    ------
    Returns: 
        Visualization of the vertices
    '''    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries([pcd], window_name, height=height, width=width)

def plot_plane(a, b, c, d, points):
    '''
    Plot a plain ax + by + cz + d = 0 and some 3D points
    '''    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Define the range of x and y values
    x = np.linspace(-1.2, 1.2, 12)
    y = np.linspace(-1.2, 1.2, 12)
    # Create a meshgrid of x and y values
    X, Y = np.meshgrid(x, y)
    # Calculate the corresponding z values using the plane equation
    if c== 0 and b == 0:
        Y, Z = np.meshgrid(x, y)
        X = -d/a
    elif c == 0:
    # If c is zero, use a different formula for Z
        Z = (-a*X - d) / b
    else:
    # If c is not zero, use the original formula for Z
        Z = (-a*X - b*Y - d) / c
    # Plot the plane as a surface
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.5)
    # Plot the points as scatter dots
    ax.scatter(points[:,0], points[:,1], points[:,2], color='red')
    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Show the plot
    plt.show()


def sequence_visualize(sequence, distance = 1.5):
    '''
    Visualize multiple meshes on the same canvas
    sequence: (np ndarray) (n,6890,3), a sequence of pointclouds, vertices for a subject
    distance: the distance from each registration in this sequence apart from each other 
    '''
    import numpy as np
    import trimesh
    import trimesh.transformations as tf

    # # Define a light grey color as a numpy array
    # light_grey = np.array([0.8, 0.8, 0.8, 1.0])

    # # Repeat the color for each vertex
    # vertex_colors = np.tile(light_grey, (6890, 1))

    params = get_params_joint_dict()
    face_array = params["f"]
    pc_array = sequence
    # Create an empty scene
    scene = trimesh.Scene()
    for i in range(pc_array.shape[0]):
        # Create a mesh from the point cloud and face arrays
        # mesh = trimesh.Trimesh(vertices=pc_array[i], faces=face_array)
        # Create a mesh from the point cloud and face arrays with vertex colors
        mesh = trimesh.Trimesh(vertices=pc_array[i], faces=face_array, face_colors = [255, 255, 255, 255])
        # mesh.face_colors = [200, 200, 200, 128]
        
        # Apply some transformation to the mesh
        # For example, translate it along the x-axis by i * 0.1 units
        # You can also use other transformations, such as rotation or scaling
        mesh.apply_transform(tf.translation_matrix([i * distance, 0, 0]))
        
        # Add the mesh to the scene
        scene.add_geometry(mesh)
    

    # Show the scene in a window
    scene.show(viewer='gl', smooth=False)    

def sequence_visualize_pro(sequence, distance = 1.5):
    import numpy as np
    import trimesh
    import trimesh.transformations as tf    
    '''
    Visualize multiple meshes on the same canvas
    sequence: (np ndarray) (row,column,6890,3), a sequence of pointclouds, vertices for a subject
    [
        train1  output1  gt1
        train2  output2  gt2
        train3  output3  gt3
        train4  output4  gt4
        train5  output5  gt5
    ]
    distance: the distance from each registration in this sequence apart from each other 
    '''
    import numpy as np
    import trimesh
    import trimesh.transformations as tf


    params = get_params_joint_dict()
    face_array = params["f"]
    pc_array = sequence
    # Create an empty scene
    scene = trimesh.Scene()
    for idx_r, row in enumerate(pc_array):
        # Create a mesh from the point cloud and face arrays
        # mesh = trimesh.Trimesh(vertices=pc_array[i], faces=face_array)
        # Create a mesh from the point cloud and face arrays with vertex colors
        for idx_c, column in enumerate(row):
            mesh = trimesh.Trimesh(vertices=column, faces=face_array, face_colors = [230, 230, 230, 255])
            # mesh = trimesh.Trimesh(vertices=column, faces=face_array)
            mesh.apply_transform(tf.translation_matrix([idx_r * distance, 0, idx_c*distance/2]))
            scene.add_geometry(mesh)

    scene.show(viewer='gl', smooth=False) 

def save_mesh(vertices, outmesh_path = './obj_files/test_mesh.obj'):
    import pickle
    src_path = "./model_processed/model_f.pkl"
    with open(src_path, 'rb') as f:
        params = pickle.load(f)
    faces = params["f"]    
    with open(outmesh_path, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def visualize_weights(weights, joint_num, height = 900, width = 720):
    '''
    weights: 6890,24, np array, a weight 
    joint_num:int, ranging from 0 to 23
    '''
    # select a joint and visualize the weights

    # Import libraries
    import numpy as np
    import open3d as o3d
    import matplotlib.cm

    params = get_params_joint_dict()
    p = params["v_template"]


    # c = params["weights"][:,joint_num]

    # Select weights for the joint no. n, find its influence on neighbour vertices
    # n = 20
    c = weights[:,joint_num]


    print(c.dtype)
    # Map the values to RGB colors using a colormap function
    colors = matplotlib.cm.jet(c).astype(np.float64)
    # print(colors[:,0:-1])
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points and colors to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(colors[:,0:-1])

    # Visualize the point cloud using Open3D
    o3d.visualization.draw_geometries([pcd], height = 900, width = 720)    

###############################################################################
################################ FUNCTIONS USAGE ##############################
###############################################################################


if __name__ == "__main__":

    ############# sample code to visualize function vert2joint_init ##########


    # import json
    # import pickle

    # joint_dict_explicit = json.load(open("./json/joint_dict_explicit.json"))
    # src_path = "./model_processed/model_f.pkl"
    # with open(src_path, 'rb') as f:
    #     params = pickle.load(f)

    # v_template = torch.from_numpy(params["v_template"])

    # verts, joints = vert2joint_init(v_template, joint_dict_explicit, get_verts = True)

    # subject_0 = torch.from_numpy(np.load("./data/sample_data/0.npy"))

    # registration = v_template
    # # registration = subject_0[0]
    # verts, joints = vert2joint_init(registration, joint_dict_explicit, get_verts = True)

    # verts, joints = verts.numpy(), joints.numpy()

    # # visualize(verts)
    # print(verts)
    



    ############# sample code to visualize function get_mean_mesh_joint ##########
    '''    
    data_dir = "./data/"
    T_mu_pose, J_mu_pose = get_mean_mesh_joint(data_dir)
    T_mu_pose, J_mu_pose = T_mu_pose.numpy(), J_mu_pose.numpy()
    
    # T_mu_pose_path = data_dir + "T_mu_pose"
    # np.save(T_mu_pose_path, T_mu_pose)

    # J_mu_pose_path = data_dir + "J_mu_pose"
    # np.save(J_mu_pose_path, J_mu_pose)
    
    T_mu_pose_path = data_dir + "J_mu_pose.npy"
    
    a = np.load(T_mu_pose_path)
    print(a.shape)
    '''

    ############# sample code to visualize function get_J_regressor ##########

    # import json
    # import pickle
    # from utils import *

    # joint_dict_explicit = json.load(open("./json/joint_dict_explicit.json"))
    # src_path = "./model_processed/model_f.pkl"
    # with open(src_path, 'rb') as f:
    #     params = pickle.load(f)

    # v_template = torch.from_numpy(params["v_template"])

    # verts, joints = vert2joint_init(v_template, joint_dict_explicit, get_verts = True)

    # J_regressor_init = get_J_regressor_init(v_template, joints)

    # np.save("./data/J_regressor_init", J_regressor_init)

    # joints_new = np.matmul(J_regressor_init, v_template)

    # print(np.allclose(joints_new, joints))

    # visualize(joints_new)

    ############# sample code to visualize sagittal plane of a pose's joints ##########

    # import h5py
    # from loss_func import *
    # fdir = r"E:\3D_HUMAN\dataset\DFAUST\registrations_f_after.hdf5"

    # with h5py.File(fdir, "r") as file:
    #     list_motion = list(file.keys())
    #     random_motion = np.random.randint(0, len(list_motion))
    #     motion = file[list_motion[random_motion]][:]
    #     random_regis = np.random.randint(0, len(motion))
    #     regis = torch.from_numpy(motion[random_regis]).type(torch.float32) 
    #     print("regis.dtype: ", regis.dtype)
    #     joints = vert2joint_init(regis)
    #     print("joints.dtype: ", joints.dtype)
    #     a, b, c, d = plane_parallel_to_oyz(joints[0])
    #     print("a.dtype: ", a.dtype)
    #     reflections = householder_transform(a, b, c, d, joints)
    #     reflections = reorder(reflections)
    #     verts_loss(joints, reflections)
    #     print(verts_loss)
    #     a, b, c, d = a.numpy(), b.numpy(), c.numpy(), d.numpy()
    #     plot_plane(a, b, c, d, reflections.numpy())

    ############# sample code to add noise to v_template ##########
    # import pickle
    # src_path = "./model_processed/model_f.pkl"
    # with open(src_path, 'rb') as f:
    #     params = pickle.load(f)

    # v_template = params["v_template"]
    # np.random.seed(47)
    # # create a random noise array of the same shape as weights
    # noise_array = np.random.normal(0, 1e-3, v_template.shape)
    # noisy_v_template = v_template + noise_array
    # save_mesh(noisy_v_template)
    # # visualize(noisy_v_template)


    ############# sample code to visualize T_hat for each subject ##########
    # import h5py

    # T_hat_path = "./init_data/T_hat_1.hdf5"
    # J_hat_path = "./init_data/J_hat_0.hdf5"
    # pose_init_path = "./init_data/theta_big.pt"
    # trans_init_path = "./init_data/trans_init_0.hdf5"

    # with h5py.File(T_hat_path, "r") as f:
    #     empty = []
    #     h5list = list(f.keys())
    #     for key in h5list:
    #         empty.append(f[key][:])   
    # empty = np.stack(empty)         
    # sequence_visualize(empty, distance=2.0)

    ############# sample code to visualize pose sequence with SMPL batch ##########
    # import h5py
    # from models.smpl_pose_batch import SMPLModel
    # device = torch.device('cuda')
    
    # # Load params:
    # pose_init_path = "./init_data/theta_big.pt"
    # trans_init_path = "./init_data/trans_init_0.hdf5"

    # pose_init = torch.load(pose_init_path)
    # pose_init = pose_init[1899:1910]

    # params = get_params_joint_dict()
    # joints = vert2joint_init(torch.from_numpy(params["v_template"]).type(torch.float32).to(device))
    
    # pose_SMPL_batch = SMPLModel(device = device)

    # with h5py.File(trans_init_path, "r") as f:
    #     empty = []
    #     h5list = list(f.keys())
    #     for key in h5list:
    #         empty.append(f[key][:])
    # empty = empty[0]
    # print(f"empty shape: {empty.shape}")
    # batch = pose_SMPL_batch(
    #     T_hat = torch.from_numpy(params["v_template"]).type(torch.float32).to(device), 
    #     J_hat = joints, 
    #     weights = torch.from_numpy(params["weights"]).type(torch.float32).to(device), 
    #     posedirs = torch.from_numpy(params["posedirs"]).type(torch.float32).to(device), 
    #     pose = pose_init.type(torch.float32).to(device), 
    #     trans = torch.from_numpy(empty).type(torch.float32).to(device)
    # )

    # batch = batch.detach().cpu().numpy()
    # sequence_visualize(batch, distance=2.0)


    # v_template_path_list = [r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50022.npy",
    # r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50025.npy",
    # r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50004.npy",
    # r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50020.npy",
    # r"E:\3D_HUMAN\dataset\DFAUST\temp\v_template_50021.npy"]

    # v_stack = []
    # for path in v_template_path_list:
    #     v_stack.append(np.load(path))
    # v_stack = np.stack(v_stack)
    # sequence_visualize(v_stack, distance=1.5)

    # import h5py
    # path = r"E:\3D_HUMAN\dataset\DFAUST\registrations_f.hdf5"
    # display = []
    # with h5py.File(path, "r") as f:
    #     key_list = list(f.keys())
    #     for i, key in enumerate(key_list):
    #         # print(key)
    #         if key == "faces":
    #             continue
    #         if i==0:    
    #             subj_id = key[:5]
    #             print("subject: ", subj_id)
    #             motion = key[5:]
    #             arr = f[key][:].transpose(2,0,1)

    #             fst_regis = arr[0]
    #             display.append(fst_regis)
    #             # print(f"subj_id: {subj_id}, motion: {motion}")
    #             # print(arr.shape[0])
    #         else:
    #             subj_id = key[:5]
    #             subj_id_pre = key_list[i-1][:5]
    #             motion = key[5:]
                
    #             if subj_id != subj_id_pre:  # enter new subject
    #                 print("subject: ", subj_id)
    #                 arr = f[key][:].transpose(2,0,1)
    #                 fst_regis = arr[0]
    #                 display.append(fst_regis)
    #                 # print(f"subj_id: {subj_id}, motion: {motion}")
    #                 # print(arr.shape[0])
    #             else:  # Still the same subject
    #                 arr = f[key][:].transpose(2,0,1)
    #                 # print(arr.shape[0])
    # display = np.stack(display)
    # sequence_visualize(display, distance=0.8)

    
    path = r"./temp/BlenderTo50004.npy"

    vertices = np.load(path)
    frame_1 = vertices[0]
    frame_2 = vertices[37]
    frame_3 = vertices[80]
    frame_4 = vertices[500]

    # full = np.stack((frame_1, frame_2, frame_3, frame_4))
    # sequence_visualize(full)

    vertices = vertices[:500]
    # params = get_params_joint_dict()
    # faces = params["f"]
    animate_point_clouds(vertices, time_sleep=0.01)