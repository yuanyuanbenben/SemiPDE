# creat simulation data

import numpy as np
import pde
import torch
from torch.utils.data import Dataset,DataLoader
import shelve

# Reaction-diffusion system
# u_t =  D u^2_x + f(u)
# Fisher-Kolmogorov Equation: f(u) = u(1-u)

# define pde class 
class ReactionDiffusionPDE(pde.PDEBase):
    def __init__(self,D=1e-3,func_f=None,bc=None):
        super().__init__()
        self.D = D
        self.func_f = (lambda u: u*(1-u)) if func_f is None else func_f
        self.bc = "auto_periodic_neumann" if bc is None else bc
        
    def evolution_rate(self,state,t=0):
        u = state[0]
        u_t = self.D*u.laplace(bc=self.bc) + self.func_f(u)
        return pde.FieldCollection([u_t])


# generate dataset
class RDModel(Dataset):
    def __init__(self, pde_model, path, curve_num=100, sample_size=100, T_range=None, delta_t=1e-3, range_grid=None, period=True, num_grid=32, seed=None, initial=None, noise_sigma=0.1, dim=1):

        if T_range is None:
            T_range = [0,3]
        if range_grid is None:
            range_grid = [-1,1]
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
                self.sample_size+1,self.dim,self.num_grid))

            for j in range(self.sample_size+1):
                if j == 0:
                    # state = pde.FieldCollection.scalar_random_uniform(
                    #     self.dim,self.grid,vmin=self.initial[0],vmax=self.initial[1],
                    #     rng=np.random.default_rng(self.seed))
                    # c1 = np.random.uniform(0.5,1)
                    # c2 = np.random.uniform(-1,1)
                    c1 = np.random.uniform(1,2)
                    c2 = np.random.uniform(0,2*np.pi)
                    state = pde.FieldCollection.from_scalar_expressions(self.grid,f"0.5*sin({c1}*(1-abs(x))+{c2})+0.5")
                    solution_collection[0,0,:] = state.data #+ np.random.normal(
                       # scale=self.noise_sigma,size=(self.dim,self.num_grid[0],self.num_grid[1]))

                else:
                    state = self.pde_model.solve(
                        state,t_range=[0,self.observed_delta_t],dt = self.delta_t,tracker=None)

                    solution_collection[j,0,:] = state.data + np.random.normal(
                        scale=self.noise_sigma,size=(self.num_grid))
            
            self.data[str(index)]= solution_collection
            states = torch.from_numpy(solution_collection).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()
            
        return {"states":states, "t": torch.arange(
            self.T_range[0],self.T_range[1]+self.observed_delta_t,self.observed_delta_t).float()}

    def __len__(self):
        return self.curve_num

# Load data for network
def create_data(path_train = "./data/train_rd", path_test = "./data/test_rd", path_utest = "./data/utest_rd", 
                D=1e-3, func_f=None, bc=None, batch_size=[16,16], train_curve_num=80, 
                test_curve_num=20, sample_size=100, T_range=None, delta_t=1e-3, range_grid=None, 
                period=True, num_grid=32, seed_train=None, 
                seed_test=None, seed_utest=None, initial=None, noise_sigma=0.1, dim=1,rand=True):

    if T_range is None:
        T_range = [0,3]
    if range_grid is None:
        range_grid = [-1,1]
    if initial is None:
        initial = [0,1]
    pde_model = ReactionDiffusionPDE(D=D,func_f=func_f,bc=bc)

    dataset_train = RDModel(
        pde_model,path_train,curve_num=train_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_train,initial=initial,noise_sigma=noise_sigma,dim=dim)

    dataset_test = RDModel(
        pde_model,path_test,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_test,initial=initial,noise_sigma=noise_sigma,dim=dim)
    
    dataset_utest = RDModel(
        pde_model,path_utest,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_utest,initial=initial,noise_sigma=0,dim=dim)

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

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)
    dataloader_utest = DataLoader(**dataloader_utest_params)

    return dataloader_train, dataloader_test, dataloader_utest