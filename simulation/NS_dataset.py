# creat simulation data

import numpy as np
import pde
import torch
from torch.utils.data import Dataset,DataLoader
import shelve

# Navier-Stokes equations
# u_t = -u \nabla u + \D_1 (u.laplacian - 1/3\nabla (div u)) - D_2 \nabla p + f_1(x) + div (f_2(x,u)(div u)I)
# p = u_x(1-|x|) + u_y(1-|y|) + u_z(1-|z|)
# f_1(x) = 0.1(sin(2pi(|x|+|y|+|z|))+ cos(2pi(|x|+|y|+|z|)))
# f_2(x,u) = (3-|x|-|y|-|z|)/(1+u^2)
# D_1 = (6e-1,7e-1,8e-1)
# D_2 = (3e-1,4e-1,5e-1)

# define pde class 
class NavierStokesPDE(pde.PDEBase):
    def __init__(self, D_1=None, D_2=None, func_f=None, func_g = None, bc=None, range_grid=None, period=None, num_grid=None):
        if D_1 is None:
            D_1 = [6e-3,7e-3,8e-3]
        if D_2 is None:
            D_2 = [3e-3,4e-3,5e-3]
        super().__init__()
        self.range_grid = range_grid
        self.period = period
        self.num_grid = num_grid
        self.grid = pde.CartesianGrid(self.range_grid,self.num_grid,periodic=self.period)
        self.bc_u = "periodic" if bc is None else bc
        self.D_1 = pde.VectorField.from_expression(self.grid,[f"{D_1[0]}",f"{D_1[1]}",f"{D_1[2]}"])
        self.D_2 = pde.VectorField.from_expression(self.grid,[f"{D_2[0]}",f"{D_2[1]}",f"{D_2[2]}"])
        if func_f is None:
            self.func_f = pde.VectorField.from_expression(self.grid,
            ["0.1*(sin(2*pi*(abs(x)+abs(y)+abs(z)))+cos(2*pi*(abs(x)+abs(y)+abs(z))))",
            "0.1*(sin(2*pi*(abs(x)+abs(y)+abs(z)))+cos(2*pi*(abs(x)+abs(y)+abs(z))))",
            "0.1*(sin(2*pi*(abs(x)+abs(y)+abs(z)))+cos(2*pi*(abs(x)+abs(y)+abs(z))))"])
        else: 
            self.func_f =  func_f

        if func_g is None:
            self.f_2 = pde.ScalarField.from_expression(self.grid,"(3-abs(x)-abs(y)-abs(z))")
            self.func_g = lambda u: (self.f_2/(1+u.to_scalar(scalar='squared_sum')) * u.divergence(bc=self.bc_u)).gradient(bc=self.bc_u)/10
        else:
            self.func_g = func_g

        self._p = pde.VectorField.from_expression(self.grid,[
            "1-abs(x)","1-abs(y)","1-abs(z)"
        ])
        
    def func_press(self,u):
        return (u.dot(self._p)).gradient(bc=self.bc_u)
    
    def evolution_rate(self,state,t=0):
        u = state
        # u.gradient(i,j,...) -> partial u_i/partial j
        return -1e-3*u.dot(u.gradient(bc=self.bc_u))+ self.D_1 * (
            u.laplace(bc=self.bc_u) - 1/3* u.divergence(self.bc_u).gradient(bc=self.bc_u)
            ) - self.D_2 * self.func_press(u) + self.func_f + self.func_g(u)

# generate dataset
class NSModel(Dataset):
    def __init__(self, pde_model, path, curve_num=100, sample_size=100, 
    T_range=None, delta_t=1e-3, range_grid=None, period=True, num_grid=16,
    seed=None, initial=None, noise_sigma=0.1, dim=3):

        if T_range is None:
            T_range = [0,2.5]
        if range_grid is None:
            range_grid = [[-1,1],[-1,1],[-1,1]]
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
                self.sample_size+1,self.dim,self.num_grid[0],self.num_grid[1],self.num_grid[2]))

            for j in range(self.sample_size+1):
                if j == 0:
                    c1,c2,c3,c5,c6,c7,c9,c10,c11 = np.random.uniform(1,2,9)
                    c4,c8,c12 = np.random.uniform(0,2*np.pi,3)
                    #state =  pde.FieldCollection.scalar_random_uniform(self.dim,self.grid,vmin=self.initial[0]+0.3,vmax=self.initial[1]-0.3,rng=np.random.default_rng(self.seed))
                    state = pde.VectorField.from_expression(self.grid,[
                        f"sin({c1}*(1-abs(x))+{c2}*(1-abs(y))+{c3}*(1-abs(z))+{c4})+1",
                        f"sin({c5}*(1-abs(x))+{c6}*(1-abs(y))+{c7}*(1-abs(z))+{c8})+1",
                        f"sin({c9}*(1-abs(x))+{c10}*(1-abs(y))+{c11}*(1-abs(z))+{c12})+1"
                        ])
                    solution_collection[0,:,:,:,:] = state.data #+ np.random.normal(
                                # scale=self.noise_sigma,size=(self.dim,self.num_grid[0],self.num_grid[1]))

                else:
                    state = self.pde_model.solve(
                        state,t_range=[0,self.observed_delta_t],dt = self.delta_t,tracker = None)

                    solution_collection[j,:,:,:,:] = state.data + np.random.normal(
                        scale=self.noise_sigma,size=(self.num_grid[0],self.num_grid[1]))

            self.data[str(index)]= solution_collection
            states = torch.from_numpy(solution_collection).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()

        return {"states":states, "t": torch.arange(
            self.T_range[0],self.T_range[1]+self.observed_delta_t,self.observed_delta_t).float()}

    def __len__(self):
        return self.curve_num

# Load data for network
def create_data(path_train = "./data/train_ns",path_test = "./data/test_ns",path_utest = "./data/utest_ns",
    D_1=5e-3,D_2=3e-3,func_f=None,func_g = None,bc=None,
    batch_size=[16,4],train_curve_num=80,test_curve_num=20,sample_size=100,T_range=[0,2.5],
    delta_t=1e-3,range_grid=[[-1,1],[-1,1],[-1,1]],period=[True,True,True],num_grid=[16,16,16],
    seed_train=None,seed_test=None,seed_utest=None,initial=[0,1],noise_sigma=0.1,dim=3,rand=True):
    # sourcery skip: default-mutable-arg

    pde_model = NavierStokesPDE(D_1=D_1,D_2 = D_2,func_f=func_f,func_g = func_g,bc=bc,range_grid=range_grid, period=period, num_grid=num_grid)

    dataset_train = NSModel(
        pde_model,path_train,curve_num=train_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_train,initial=initial,noise_sigma=noise_sigma,dim=dim)

    dataset_test = NSModel(
        pde_model,path_test,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_test,initial=initial,noise_sigma=noise_sigma,dim=dim)
    
    dataset_utest = NSModel(
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