import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader

# dataset
class Datagen(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data = np.load(path)
        self.curve_num = self.data.shape[0]
        # if self.curve_num == 15:
        #     self.case_index = torch.tensor([1,1,1,2,2,3,3,3,4,4,5,5,6,6,6]).float()
        # else:
        #     self.case_index = torch.tensor([2,4,5]).float()
            
    def __getitem__(self,index):
        # get the index curve with shape batchsize * timegrid * spacegrid
        # here is 1 * 5 * 38
        return {'states': torch.tensor(self.data[index,:,:]).float(),
                't': torch.tensor([0,12,24,36,48]).float(),}
                # 'case':self.case_index[index]}

    def __len__(self):
        return self.curve_num
    
def creat_data(path,batch_size,device=None):
    data = Datagen(path)
    # if device is None:
    dataloader_params = {
        'dataset'    : data,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : True,
    }
    # else:
    #     dataloader_params = {
    #         'dataset'    : data,
    #         'batch_size' : batch_size,
    #         'num_workers': 0,
    #         'pin_memory' : False,
    #         'drop_last'  : False,
    #         'shuffle'    : True,
    #         'generator' : torch.Generator(device=device)
    #     }
    return DataLoader(**dataloader_params)