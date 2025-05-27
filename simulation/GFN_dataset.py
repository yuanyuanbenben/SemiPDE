# creat simulation data

import numpy as np
import pde
import torch
from torch.utils.data import Dataset,DataLoader
import shelve

# General FitzHugh–Nagumo equations
# u_t = D_1 Laplace u + f(u,w), f(u,w) = u (1 - u) (u - a) - w
# w_t = D_2 Laplace w + g(u,w), g(u,w) = \epsilon (u - \gamma w)
# D_1,D_2 >= 0, \epsilon,\gamma > 0

# define pde class 
class GeneralFitzhughNagumoPDE(pde.PDEBase):
    def __init__(self,D_1=1e-3,D_2=5e-3,func_f=None,func_g=None,a=0.1,epsilon=1,gamma=1,bc=None):
        super().__init__()
        self.D_1 = D_1
        self.D_2 = D_2
        if func_f is None:
            self.a = a
            self.func_f = lambda u,w: u*(1-u)*(u-self.a) - w
        else:
            self.func_f = func_f
        if func_g is None:
            self.epsilon = epsilon
            self.gamma = gamma
            self.func_g = lambda u,w: self.epsilon*(u-self.gamma*w)
        else:
            self.func_g = func_g
        if bc is None:
            self.bc_u = "auto_periodic_neumann"
            self.bc_w = "auto_periodic_neumann"
        else: 
            self.bc_u = bc["u"]
            self.bc_w = bc["w"]
        
    def evolution_rate(self,state,t=0):
        u,w = state
        u_t = self.D_1*u.laplace(bc=self.bc_u) + self.func_f(u,w)
        w_t = self.D_2*w.laplace(bc=self.bc_w) + self.func_g(u,w)
        return pde.FieldCollection([u_t,w_t])
    

# generate dataset
class GFNModel(Dataset):
    def __init__(self, pde_model, path, curve_num=100, sample_size=100, T_range=None, delta_t=1e-3, range_grid=None, period=None, num_grid=None, seed=None, initial=None, noise_sigma=0.1, dim=2):

        if T_range is None:
            T_range = [0,3]
        if range_grid is None:
            range_grid = [[-1,1],[-1,1]]
        if period is None:
            period = [True,True]
        if num_grid is None:
            num_grid = [32,32]
        if initial is None:
            initial = [0,1]
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
        self.initial = initial
        self.noise_sigma = noise_sigma
        self.observed_delta_t = (self.T_range[1]-self.T_range[0])/self.sample_size
        self.rate = self.observed_delta_t//self.delta_t
        self.dim = dim
        if self.seed is None:
            self.seed = 20221104
        np.random.seed(self.seed)

        self.grid = pde.CartesianGrid(self.range_grid,self.num_grid,periodic=self.period)

    
    def __getitem__(self,index):
        if self.data.get(str(index)) is None:
            solution_collection = np.ndarray(shape=(
                self.sample_size+1,self.dim,self.num_grid[0],self.num_grid[1]))

            for j in range(self.sample_size+1):
                if j == 0:
                    c1 = np.random.uniform(1,2)
                    c2 = np.random.uniform(1,2)
                    c3 = np.random.uniform(0,2*np.pi)
                    c4 = np.random.uniform(1,2)
                    c5 = np.random.uniform(1,2)
                    c6 = np.random.uniform(0,2*np.pi)
                    state1 = pde.ScalarField.from_expression(self.grid, f"0.5*sin({c1}*(1-abs(x))+{c2}*(1-abs(y))+{c3})+0.5")
                    state2 = pde.ScalarField.from_expression(self.grid, f"0.5*sin({c4}*(1-abs(x))+{c5}*(1-abs(y))+{c6})+0.5")
                    state = pde.FieldCollection([state1,state2])
                    #state = pde.FieldCollection.scalar_random_uniform(
                        #self.dim,self.grid,vmin=self.initial[0],vmax=self.initial[1],
                        #rng=np.random.default_rng(self.seed))

                    solution_collection[j,:,:,:] = state.data #+ np.random.normal(
                        # scale=self.noise_sigma,size=(self.dim,self.num_grid[0],self.num_grid[1]))

                else:
                    state = self.pde_model.solve(
                        state,t_range=[0,self.observed_delta_t],dt = self.delta_t,tracker = None)

                    solution_collection[j,:,:,:] = state.data + np.random.normal(
                        scale=self.noise_sigma,size=(self.dim,self.num_grid[0],self.num_grid[1]))
            
            self.data[str(index)]= solution_collection
            states = torch.from_numpy(solution_collection).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()
            
        return {"states":states, "t": torch.arange(
            self.T_range[0],self.T_range[1]+self.observed_delta_t,self.observed_delta_t).float()}

    def __len__(self):
        return self.curve_num

# Load data for network
def create_data(path_train = "./data/train", path_test = "./data/test", path_utest = "./data/utest", 
                D_1=1e-3, D_2=5e-3, func_f=None, func_g=None, a=0.1, epsilon=1, gamma=1, bc=None,
                batch_size=[40,10], train_curve_num=80, test_curve_num=20, 
                sample_size=100, T_range=None, delta_t=1e-3, range_grid=None, period=None, 
                num_grid=None, seed_train=None, seed_test=20221106, seed_utest=None, initial=None, noise_sigma=0.1, dim=2,device = None,rand=True):

    if T_range is None:
        T_range = [0,3]
    if range_grid is None:
        range_grid = [[-1,1],[-1,1]]
    if period is None:
        period = [True,True]
    if num_grid is None:
        num_grid = [32,32]
    if initial is None:
        initial = [0,1]
        
    pde_model = GeneralFitzhughNagumoPDE(
        D_1=D_1,D_2=D_2,func_f=func_f,func_g=func_g,
        a=a,epsilon=epsilon,gamma=gamma,bc=bc)

    dataset_train = GFNModel(
        pde_model,path_train,curve_num=train_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_train,initial=initial,noise_sigma=noise_sigma,dim=dim)

    dataset_test = GFNModel(
        pde_model,path_test,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_test,initial=initial,noise_sigma=noise_sigma,dim=dim)
    
    dataset_utest = GFNModel(
        pde_model,path_utest,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_utest,initial=initial,noise_sigma=0,dim=dim)
    
    if device is None:
        dataloader_train_params = {
            'dataset'    : dataset_train,
            'batch_size' : batch_size[0],
            'num_workers': 0,
            'pin_memory' : rand,
            'drop_last'  : False,
            'shuffle'    : rand,
        }

        dataloader_test_params = {
            'dataset'    : dataset_test,
            'batch_size' : batch_size[1],
            'num_workers': 0,
            'pin_memory' : True,
            'drop_last'  : False,
            'shuffle'    : False,
        }
        
        dataloader_utest_params = {
            'dataset'    : dataset_utest,
            'batch_size' : batch_size[1],
            'num_workers': 0,
            'pin_memory' : True,
            'drop_last'  : False,
            'shuffle'    : False,
        }
    else:
        dataloader_train_params = {
            'dataset'    : dataset_train,
            'batch_size' : batch_size[0],
            'num_workers': 0,
            'pin_memory' : False,
            'drop_last'  : False,
            'shuffle'    : True,
            'generator' : torch.Generator(device=device)
        }

        dataloader_test_params = {
            'dataset'    : dataset_test,
            'batch_size' : batch_size[1],
            'num_workers': 0,
            'pin_memory' : False,
            'drop_last'  : False,
            'shuffle'    : False,
            'generator' : torch.Generator(device=device)
        }
        
        dataloader_utest_params = {
            'dataset'    : dataset_utest,
            'batch_size' : batch_size[1],
            'num_workers': 0,
            'pin_memory' : True,
            'drop_last'  : False,
            'shuffle'    : False,
        }

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)
    dataloader_utest = DataLoader(**dataloader_utest_params)
    return dataloader_train, dataloader_test, dataloader_utest

# def pinn_create_data(path_train = "./data/train", path_utest = "./data/utest", 
#                 D_1=1e-3, D_2=5e-3, func_f=None, func_g=None, a=0.1, epsilon=1, gamma=1, bc=None,
#                 batch_size=[40,10], train_curve_num=80, test_curve_num=20, 
#                 sample_size=100, T_range=None, delta_t=1e-3, range_grid=None, period=None, 
#                 num_grid=None, seed_train=None, seed_utest=None, initial=None, noise_sigma=0.1, dim=2,device = None,rand=True):

#     if T_range is None:
#         T_range = [0,3]
#     if range_grid is None:
#         range_grid = [[-1,1],[-1,1]]
#     if period is None:
#         period = [True,True]
#     if num_grid is None:
#         num_grid = [32,32]
#     if initial is None:
#         initial = [0,1]
        
#     pde_model = GeneralFitzhughNagumoPDE(
#         D_1=D_1,D_2=D_2,func_f=func_f,func_g=func_g,
#         a=a,epsilon=epsilon,gamma=gamma,bc=bc)

#     dataset_train = GFNModel(
#         pde_model,path_train,curve_num=train_curve_num,sample_size=sample_size,
#         T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
#         num_grid=num_grid,seed=seed_train,initial=initial,noise_sigma=noise_sigma,dim=dim)
    
#     dataset_utest = GFNModel(
#         pde_model,path_utest,curve_num=test_curve_num,sample_size=sample_size,
#         T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
#         num_grid=num_grid,seed=seed_utest,initial=initial,noise_sigma=0,dim=dim)
    
#     if device is None:
#         dataloader_train_params = {
#             'dataset'    : dataset_train,
#             'batch_size' : batch_size[0],
#             'num_workers': 0,
#             'pin_memory' : False,
#             'drop_last'  : False,
#             'shuffle'    : rand,
#         }
        
#         dataloader_utest_params = {
#             'dataset'    : dataset_utest,
#             'batch_size' : batch_size[1],
#             'num_workers': 0,
#             'pin_memory' : False,
#             'drop_last'  : False,
#             'shuffle'    : rand,
#         }
#     else:
#         dataloader_train_params = {
#             'dataset'    : dataset_train,
#             'batch_size' : batch_size[0],
#             'num_workers': 0,
#             'pin_memory' : False,
#             'drop_last'  : False,
#             'shuffle'    : rand,
#             'generator' : torch.Generator(device=device)
#         }

#         dataloader_utest_params = {
#             'dataset'    : dataset_utest,
#             'batch_size' : batch_size[1],
#             'num_workers': 0,
#             'pin_memory' : False,
#             'drop_last'  : False,
#             'shuffle'    : rand,
#         }

#     dataloader_train = DataLoader(**dataloader_train_params)
#     dataloader_utest = DataLoader(**dataloader_utest_params)
#     return dataloader_train,dataloader_utest

