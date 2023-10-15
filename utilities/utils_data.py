import h5py
import os
import numpy as np
import torch
def loss(a,b):
    return np.sum((a-b)**2.0)

def downsamp_data(fdir, tdir, delta = 2):
    '''
    This function reduces the number of duplicated registrations (regis) or 
    slightly different from each other an amount of delta
    in a sequence of regises in a motion and remove key "faces". Sometimes a motion
    contains a lot a the actor standing still or the motion was captured
    by a high FPS capturing device. 
    The idea behind this code is using a threshold delta and an anchor.
    First the anchor is attached to the first regis. And it goes through
    all the regis until the difference between them (in this case, L2 norm is used) 
    is larger than delta then the next anchor is defined by this current anchor. 
    All anchors are saved and stored to a new h5py file.
    Downsampled data is stored in the same dir as the origin data

    Note that the saved data is transposed .transpose(2,0,1)

    Args: 
        fdir: str, the file path of the .h5 data
        delta: the threshold for the anchor
    Returns:
        None
    '''
    pre_data_path = fdir

    with h5py.File(pre_data_path, "r") as f:
        motion_list = list(f.keys())
        motion_list = motion_list[:-1]   # remove the "faces" key
        arr_dict = dict()
        for motion in motion_list:
            arr_dict[motion] = []

        for motion in motion_list:
            motion_arr = f[motion][:].transpose([2,0,1])
            motion_len = len(motion_arr)
            total = 0
            anchor_index = 0
            arr_dict[motion].append(motion_arr[anchor_index])
            for i in range(motion_len):            
                diff = loss(motion_arr[anchor_index], motion_arr[i])
                if diff > delta:            
                    total += 1            
                    # print(f"Total: {total}, anchor registration {anchor_index} diff to registration {i}: {diff}")
                    anchor_index = i
                    arr_dict[motion].append(motion_arr[anchor_index])            
            # print(f"Original shape: {motion_arr.shape}, after reduction: {total} ")

        after_data_path = tdir

        with h5py.File(after_data_path, "w") as file:  # "w" is write
            for key, value in arr_dict.items():
                # Create a dataset with the same name as the key and the value as the data
                file.create_dataset(key, data=value)

        # Close the file object
        file.close()


############### Down sample but balance out the data for each movement
def loss_torch(a,b):
    return torch.sum((a-b)**2)
device = torch.device('cuda')

def cal_num_ancor(epsilon_, verts):
    '''
    Args:
        epsilon_: the amount of difference to change ancor
        verts: torch: nx6890x3, a sequence of registration
    Returns:
        num_ancor: the len of the ancors array
        ancors: the downsamp registrations
    '''
    num_ancor = 0.
    
    ancors = []
    for i, regis in enumerate(verts):
        if i == 0:
            ancor = regis
            num_ancor += 1.
            ancors.append(ancor)
            i_fin = i
        else:
            diff = loss_torch(ancor, regis)
            if diff > epsilon_:
                ancor = regis
                num_ancor += 1.
                ancors.append(ancor)
                i_fin = i
    return num_ancor, ancors, i_fin

def find_ancors(goal, verts, start, stop, delta=1):
    '''
    Args:
        goal: integer: desired number of downsamp registration
        verts: torch: nx6890x3, a sequence of registration
        delta: for the linspace's smoothness
    Returns:
        a numpy array of shape goalx6890x3 containing the downsamp sequence
    '''
    # stop = 100
    # start = 0
    num = 2*stop + 1
    iter_list = np.linspace(start = start, stop = stop, num = num)
    # print(iter_list)
    for eps in iter_list:
        num_ancor, ancors, i_fin = cal_num_ancor(eps, verts)
        # print('epsilon: {}, num ancors: {}'.format(eps, num_ancor))

        bias = np.abs(goal - num_ancor)
            
        if bias < delta:
            # print(f"sucessfully downsamp, eps: {eps}, num_regis: {num_ancor}")
            break
    # if eps==stop:
    #     print(f"        failed downsamp, eps: {eps}, num_regis: {num_ancor}")
    
    try:
        ancors = torch.stack(ancors).detach().cpu().numpy()
    except:
        ancors = torch.stack(ancors).numpy()
    return ancors, i_fin

'''
Usage for down samp balance on DFaust dataset for training set
Avoid movement one leg raise and first half of running

path_m = r"E:\3D_HUMAN\dataset\DFAUST\registrations_m.hdf5"
path_f = r"E:\3D_HUMAN\dataset\DFAUST\registrations_f.hdf5"

maleList = [
'50002',
'50007',
'50009',
'50026',
'50027',  
]
maleDict = dict()
femaleList = [
'50004',
'50020',
'50021',
'50022',
'50025',
]
femaleDict = dict()
fileMa = h5py.File(path_m, 'r')
keyListMa = list(fileMa.keys())
fileFe = h5py.File(path_f, 'r')
keyListFe = list(fileFe.keys())

### Male
mBySubj = dict()
numMotionM = {
'50002':23,
'50007':25,
'50009':30,
'50026':23,
'50027':25,
}
keyListM = list(fileMa.keys())
keyListM = keyListM[:-1] # a list for sid + pid, -1 to remove 'faces'

for subjMo in keyListM:
    subjID = subjMo[:5]
    pID = subjMo[6:]
    
    if pID == 'one_leg_loose':
        print("Skip")
        continue
    elif pID == 'running_on_spot':
        print("Take the second half")
        arr = fileMa[subjMo][:].transpose(2, 0, 1)
        arr = torch.from_numpy(arr).to(device)
        half = int(len(arr)/2)
        # Take the first half to test, leave the rest for train
        arr = arr[half:]
        downsampArr, i_fin = find_ancors(goal=numMotionM[subjID], start=0, stop=100, verts=arr, delta=2)
    
    else:
        arr = fileMa[subjMo][:].transpose(2, 0, 1)
        arr = torch.from_numpy(arr).to(device)
        downsampArr, i_fin = find_ancors(goal=numMotionM[subjID], start=0, stop=100, verts=arr, delta=2)
    mBySubj[subjMo] = downsampArr
    print(f"subject {subjID}, pID: {pID} last one: {i_fin}/{len(arr)} num ancor {len(downsampArr)}")

    
### Female
fBySubj = dict()
numMotionF = {
'50004':23,
'50020':25,
'50021':30,
'50022':25,
'50025':25,
}
keyListF = list(fileFe.keys())
keyListF = keyListF[:-1] # a list for sid + pid

for subjMo in keyListF:
    subjID = subjMo[:5]
    pID = subjMo[6:]
    
    if pID == 'one_leg_loose':
        print("Skip")
        continue
    elif pID == 'running_on_spot':
        print("Take the second half")
        arr = fileFe[subjMo][:].transpose(2, 0, 1)
        arr = torch.from_numpy(arr).to(device)
        half = int(len(arr)/2)
        # Take the first half to test, leave the rest for train
        arr = arr[half:]
        downsampArr, i_fin = find_ancors(goal=numMotionF[subjID], start=0, stop=100, verts=arr, delta=2)
    
    else:
        arr = fileFe[subjMo][:].transpose(2, 0, 1)
        arr = torch.from_numpy(arr).to(device)
        downsampArr, i_fin = find_ancors(goal=numMotionF[subjID], start=0, stop=100, verts=arr, delta=2)
    fBySubj[subjMo] = downsampArr
    print(f"subject {subjID}, pID: {pID} last one: {i_fin}/{len(arr)} num ancor {len(downsampArr)}")


'''


def regis_by_subject(path, tdir):
    '''
    path=r"E:\3D_HUMAN\dataset\DFAUST\registrations_m_after.hdf5"
    tdir=r"E:\3D_HUMAN\dataset\DFAUST\downsamp_data\DFAUST_by_subject_f.hdf5"
    Convert a h5py file with key as id_motion to a h5py with key as subject's id. 
    All case key's values are registrations.
    Args:
        path: path to hdf5 files after downsampled by func downsamp_data above
        tdir: destination to save hdf5 file
    Returns:
        None
    '''
    filename = path

    with h5py.File(filename, "r") as f:  # "r" is read only
        motion_list = list(f.keys())   # motion_list = ['50004_chicken_wings', '50004_hips', '50004_jiggle_on_toes',...]

    subject_no_list = [motion[:5] for motion in motion_list]  # Keep the ids only
    subject_list = list(set(subject_no_list))  # Removes duplicates
    print("subject_list: ", subject_list)
    # subject_list.remove('faces')   # subject_list = ['50025', '50022', '50004', '50021', '50020']

    # Create a dict with keys as subject ids, key's value as the motions that subjects did
    subject_dict = dict()
    for subject in subject_list:
        subject_dict[subject] = []

    for motion in motion_list:
        for subject in subject_list:
            if motion[:5] == subject:
                subject_dict[subject].append(motion)

    arr_dict = dict()
    for subject in subject_list:
        arr_dict[subject] = []

    with h5py.File(filename, "r") as f:
        for subject in subject_dict:  # subject = "50004"
            print(f"Subject id: {subject}")
            for motion_sub in subject_dict[subject]:  # motion_sub = "50004_hips"
                motion_arr = f[motion_sub][:]
                print(f"motion_arr shape: {motion_arr.shape}")
                # motion_arr = f[motion_sub][()].transpose([2,0,1])  # shape 223,6890,3  # Maybe can substitute [()] with "[:]"
                # print("Regis for subject ", subject, " in motion ", motion_sub, " has shape ", motion_arr.shape)
                # motion_arr_0 = motion_arr[0]  # shape 6890,3
                arr_dict[subject].append(motion_arr)  # arrdict['50004'] append[n,6890, 3]
            # arr_dict[subject] is an array containing multiple nx6890x3 arrays    
            arr_dict[subject] = np.concatenate(arr_dict[subject], axis = 0)    

    with h5py.File(tdir, "w") as file:  # "w" is write
        # Loop through the dictionary keys and values
        for key, value in arr_dict.items():
            # Create a dataset with the same name as the key and the value as the data
            file.create_dataset(key, data=value)

    # Close the file object
    file.close()    

if __name__ == "__main__":
    # downsamp_data(fdir=r"E:\3D_HUMAN\dataset\DFAUST\registrations_f.hdf5",
    #               tdir=r"E:\3D_HUMAN\dataset\DFAUST\registrations_f_delta_80.hdf5"
    #               , delta=80)
    regis_by_subject(path=r"E:\3D_HUMAN\Code\train_SMPL_Final\temp\train_downsamp.h5py",
                     tdir=r"E:\3D_HUMAN\Code\train_SMPL_Final\temp\train_by_subj.h5py")
