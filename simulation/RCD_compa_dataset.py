# creat simulation data

import numpy as np
import pde
import torch
from torch.utils.data import Dataset,DataLoader
import shelve

# ReactionConvectionDiffusionPDE
# u_t = D_1 u_xx + D_2 u_x + f(u)
# D_1 = 1 D_2 = 0.1 
# u(t,0) = 0, u(0,x) = (1+0.1(20-x)^2)^{-1}
# t [1,20],  x [1,40]

# define pde class 
class ReactionConvectionDiffusionPDE(pde.PDEBase):
    def __init__(self, D_1= 1, D_2 = 0.1, func_f = None, bc=None, range_grid=None, period=None, num_grid=None):
        self.func_f = (lambda u: 0.1*u) if func_f is None else func_f
        super().__init__()
        self.range_grid = range_grid
        self.period = period
        self.num_grid = num_grid
        self.grid = pde.CartesianGrid(self.range_grid,self.num_grid,periodic=self.period)
        self.bc_u = "periodic" if bc is None else bc
        self.D_1 = D_1
        self.D_2 = D_2
        
    
    def evolution_rate(self,state,t=0):
        u = state
        return self.D_1 * u.laplace(bc = self.bc_u) + self.D_2 * u.gradient(bc = self.bc_u).to_scalar() + self.func_f(u)

# generate dataset
class RCDModel(Dataset):
    def __init__(self, pde_model, path, curve_num=100, sample_size=100, 
    T_range=None, delta_t=1e-3, range_grid=None, period=True, num_grid=16,
     seed=None, noise_sigma=0.1, dim=1):

        if T_range is None:
            T_range = [0,20]
        if range_grid is None:
            range_grid = [[0,40]]
        super().__init__()
        self.data = shelve.open(path)
        self.pde_model = pde_model
        self.curve_num = curve_num
        self.sample_size = sample_size
        self.T_range = T_range
        self.delta_t = delta_t
        self.range_grid = range_grid
        self.period = period
        self.num_grid = num_grid
        self.seed = seed
        self.noise_sigma = noise_sigma
        self.dim = dim
        self.observed_delta_t = (self.T_range[1]-self.T_range[0])/self.sample_size
        self.rate = self.observed_delta_t//self.delta_t
        if self.seed is None:
            self.seed = 20221104
        np.random.seed(self.seed)
        self.grid = pde.CartesianGrid(self.range_grid,self.num_grid,periodic=self.period)
    
    def __getitem__(self,index):
        if self.data.get(str(index)) is None:
            solution_collection = np.ndarray(shape=(
                self.sample_size+1,self.dim,self.num_grid[0]))
            for j in range(self.sample_size+1):
                if j == 0:
                    state = pde.ScalarField.from_expression(self.grid,'1/(1+0.1*(20-x)**2)')
                    solution_collection[0,:,:] = state.data #+ np.random.normal(
                                   # scale=self.noise_sigma,size=(self.dim,self.num_grid[0],self.num_grid[1]))
                else:
                    state = self.pde_model.solve(
                        state,t_range=[0,self.observed_delta_t],dt = self.delta_t,tracker = None)

                    solution_collection[j,:,:] = state.data + np.random.normal(
                        scale=self.noise_sigma,size=(self.num_grid[0]))

            self.data[str(index)]= solution_collection
            states = torch.from_numpy(solution_collection).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()

        return {"states":states, "t": torch.arange(
            self.T_range[0],self.T_range[1]+self.observed_delta_t,self.observed_delta_t).float()}

    def __len__(self):
        return self.curve_num

# Load data for network
def create_data(path_train = "./data/train_rcd",path_test = "./data/test_rcd",
    D_1=5e-3,D_2=3e-3,func_f=None,bc=None,
    batch_size=[16,4],train_curve_num=80,test_curve_num=20,sample_size=100,T_range=[0,2.5],
    delta_t=1e-3,range_grid=[[-1,1],[-1,1],[-1,1]],period=[True,True,True],num_grid=[16,16,16],
    seed_train=None,seed_test=20221106,noise_sigma=0.1,dim=3):
    # sourcery skip: default-mutable-arg

    pde_model = ReactionConvectionDiffusionPDE(D_1=D_1,D_2 = D_2,func_f=func_f,
    bc=bc,range_grid=range_grid, period=period, num_grid=num_grid)

    dataset_train = RCDModel(
        pde_model,path_train,curve_num=train_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_train,noise_sigma=noise_sigma,dim=dim)

    dataset_test = RCDModel(
        pde_model,path_test,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_test,noise_sigma=noise_sigma,dim=dim)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : batch_size[0],
        'num_workers': 0,
        'pin_memory' : False,
        'drop_last'  : False,
        'shuffle'    : False,
        'generator'  : torch.Generator(device='cuda'),
    }
    
    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : batch_size[1],
        'num_workers': 0,
        'pin_memory' : False,
        'drop_last'  : False,
        'shuffle'    : False,
        'generator'  : torch.Generator(device='cuda'),
    }

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test