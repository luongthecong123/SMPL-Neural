import torch
from torch.utils.data import Dataset

class Subject_Regis(Dataset):
    '''
    The whole dataset is loaded to cpu RAM
    The batch is loaded to GPU (Nvidia cuda)
    '''
    def __init__(self, id_list, regises):
        """
        Args:
            
            id_list: a list contains the id for each regis in regises
            regises: an np array contains all 
        """
        # self.ids = [int(i) for i in id_list]
        self.device = torch.device('cuda')
        self.ids = list(map(int, id_list))
        self.ids = id_list
        # print(self.ids)
        self.regises = torch.from_numpy(regises)  # Transform into tensor on cpu
    def __len__(self):
        return len(self.ids)
# __getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
    def __getitem__(self, idx):
        regis = self.regises[idx].to(self.device)
        id = self.ids[idx]
        return regis, id, idx

'''    
    data = Subject_Regis(id_list, regis_arr)

    # Check your cpu RAM for num_workers
    # If you use multiple workers in your DataLoader, each worker will load a batch 
    # in the background using multiprocessing while the GPU is busy.
    # In this case, the entire dataset is loaded in Dataset __init__, so
    # num workers should = 0 to avoid increasing memory significantly
    # If registraions are stored individually on hard drive, then num_workers can be ultilized
    # https://discuss.pytorch.org/t/what-data-does-each-worker-process-hold-does-it-hold-the-full-dataset-object-or-only-a-batch-of-it/160136
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=2)

    hehe = iter(dataloader)

    regises, ids = next(hehe)
    print(ids)
    sequence_visualize(regises.numpy())
    '''