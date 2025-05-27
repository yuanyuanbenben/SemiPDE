# creat simulation data

import numpy as np
import pde
import torch
from torch.utils.data import Dataset,DataLoader
import shelve
import copy

# ReactionConvectionDiffusionPDE
# u_t = D_1 u_xx + D_2 u_x + f(u)
# D_1 = 1 D_2 = 0.1 
# u(t,0) = 0, u(0,x) = (1+0.1(20-x)^2)^{-1}
# t [1,20],  x [1,40]

# class PINN_data(Dataset):
#     def __init__(self, path):
#         super().__init__()
#         self.state = torch.from_numpy(shelve.open(path)['0']['states']).float().reshape(-1,1)
#         self.len = self.state.size()[0]
#         self.position = torch.from_numpy(shelve.open(path)['0']['position']).float().permute(0,2,1).contiguous().view(self.len,2)
    
#     def __getitem__(self,index):
#         return {"states":self.state[index,:], 
#                 "position":self.position[index,:]}

#     def __len__(self):
#         return self.len
    
class PINN_data(Dataset):
    def __init__(self,noise,seed=251785105):
        super().__init__()
        data = np.ndarray((21,43))
        data[0,1:42] = [1/(1+0.1*(20-x)**2) for x in np.linspace(0,40,41,endpoint=True)]
        data[0,0] = data[0,41]
        data[0,42] = data[0,1]
        np.random.seed(seed)
        for t in range(1,21):
            before_u = copy.deepcopy((data[t-1,:]))
            for dt in range(10):
                grad_u = copy.deepcopy(before_u[2:] - before_u[:41])/2
                lap_u = copy.deepcopy(before_u[2:] - 2*before_u[1:42] + before_u[:41])
                now_u = copy.deepcopy(before_u[1:42] + (lap_u + 0.1 * grad_u + 0.1 * before_u[1:42]) * 0.1)
                before_u[1:42] = copy.deepcopy(now_u)
                before_u[0] = copy.deepcopy(now_u[-1])
                before_u[-1] = copy.deepcopy(now_u[0])
            data[t,:] = copy.deepcopy(before_u)
        self.data = data[:,np.int0(np.linspace(1,40,40,endpoint=True))]
        # print(self.data)
        self.data[1:,:] = self.data[1:,:] + np.random.normal(scale=noise,size=(20,40))
        # self.boundary1 = torch.from_numpy(np.array([[t/10-1,-2.0] for t in range(21)])).float()
        # self.boundary2 = torch.from_numpy(np.array([[t/10-1,2.0] for t in range(21)])).float()
        self.position = torch.from_numpy(np.array([[t/10-1,x/10-2] for t in range(21) for x in range(1,41)])).float()
        # self.in_position = torch.from_numpy(np.array([[t/10-1,x/10-2] for t in range(20) for x in range(1,40)])).float()
        self.data = torch.from_numpy(self.data).float().reshape(-1,1)
        #print(self.data)
        #print(self.position)
        self.len = self.data.size()[0]
        
    def __getitem__(self,index):
        return {'states':self.data[index,:], 
                'position':self.position[index,:],
                # 'boundary1':self.boundary1[index%21,:],
                # 'boundary2':self.boundary2[index%21,:],
                # 'in_position':self.in_position[index%(20*39),:],
                }
        
    def __len__(self):
        return self.len

def create_pinn_data(D_1=5e-3,D_2=3e-3,func_f=None,bc=None,
    batch_size=[16,4],train_curve_num=80,test_curve_num=20,sample_size=100,T_range=[0,2.5],
    delta_t=1e-3,range_grid=[[-1,1],[-1,1],[-1,1]],period=[True,True,True],num_grid=[16,16,16],
    seed_train=None,seed_test=20221106,noise_sigma=0.1,dim=3):
    # sourcery skip: default-mutable-arg


    dataset_train = PINN_data(noise=noise_sigma)

    dataset_test = PINN_data(noise = noise_sigma)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : batch_size[0],
        'num_workers': 0,
        'pin_memory' : False,
        'drop_last'  : False,
        'shuffle'    : True,
        'generator'  : torch.Generator(device='cuda'),
    }
    
    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : batch_size[1],
        'num_workers': 0,
        'pin_memory' : False,
        'drop_last'  : False,
        'shuffle'    : True,
        'generator'  : torch.Generator(device='cuda'),
    }

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test