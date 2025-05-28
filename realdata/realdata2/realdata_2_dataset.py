import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

# dataset
class Datagen(Dataset):
    def __init__(self, path,curve_para,repeat_curve):
        super().__init__()
        self.path = path
        self.a = curve_para['a']
        self.b = curve_para['b']
        self.c = curve_para['c']
        self.init_phase = curve_para['init_phase']
        self.repeat_curve = repeat_curve
        data = np.load(path)
        currents,repeat,h,d = data.shape
        # data = data.transpose(0,1,3,2)
        # self.data = data.reshape((currents*repeat*d,h))
        self.data = np.ndarray((currents*repeat*6,81,d))
        for i in range(currents):
            for j in range(repeat):
                for k in range(6):
                    self.data[(i*repeat*6+j*6 + k),:,:] = data[i,j,(80*k):(80*k+81),:]
        # x = np.linspace(0,1.6,81,endpoint=True)
        # plt.plot(x,self.data[0,:,0],x,self.data[1,:,0],x,self.data[2,:,0],x,self.data[3,:,0],x,self.data[4,:,0],x,self.data[5,:,0])
        # plt.savefig('./pic/plots.png')
        self.x_position = torch.tensor([[x for x in curve_para['position']] for y in range(currents*repeat*6)]).float()
        self.curve_num = self.data.shape[0]
        # _aug = [wave_height*2*np.pi/wave_time*np.sqrt(0.05**2 -((2*x-x_space+1)*wave_height/(x_space-1))**2) for x in range(x_space)]
        # self.aug = np.zeros(x_space)
        # for i in range(1,x_space):
        #     self.aug[i] = self.aug[i-1] + (_aug[i]+_aug[i-1])*wave_height/(x_space-1)
        # self.aug = self.aug - self.aug[x_space//2]
        
    def aug(self):
        # a sin (2pi/0.096 x + b) + c
        init_tensor = self.a * torch.sin(self.init_phase + self.b) + self.c
        return init_tensor
        
    def __getitem__(self,index):
        # get the index curve with shape batchsize * timegrid
        # here is 1 * 481
        index = index+6*self.repeat_curve
        states = self.data[index,:,:]
        aug_int = self.aug()
        # print(aug_int)
        # aug_init = init_states + self.aug
        return {'states': torch.tensor(states).float(),
                't': torch.tensor(np.linspace(0,1.6,81,endpoint=True)).float(),
                'x_position': self.x_position[index,:],
                'initial':aug_int}

    def __len__(self):
        return 6
    
def creat_data(path, batch_size,curve_para,repeat_curve):
    data = Datagen(path,curve_para,repeat_curve)
    dataloader_params = {
        'dataset'    : data,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : True,
    }
    return DataLoader(**dataloader_params)