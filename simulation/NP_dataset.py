# creat simulation data

import numpy as np
import pde
import torch
from torch.utils.data import Dataset,DataLoader
import shelve

# Nernst-Planck equation
# u_t = D_1 u.laplacian + D_2 (u^2 - E div u) - div(v(x)u)
# E.laplacian = - u
# u mol m-2 s-1
# consider CuCl2-EG for cu 2+ to cu 1+ with T=343K D = 7e-11 m2 s-1
# z= 1 
# e = 1.60217663 × 10-19 A s
# kb = 1.380649 × 10-23 m2 kg s-2 K-1
# epsilon =  710 × 10−12 m-3kg-1s2c2  (for water)
# D_1 = 7e-11 m2 s-1 D_2 = D_1/epsilon * z2e2/kb T = 5.33e-19
# consider a scale transform from [0,1e-8] to [0,1] and u to 10^8 u for numerical stability
# use D_1 = 7e-3 D_2 = 5.33e-3, E = (1-u(1-|x|))/2,1-(u(1-|y|)/2)), v(x) = ((1-|x|)^2,(1-|y|)^2)
# from t= 0 to 1


# define pde class 
class NernstPlanckPDE(pde.PDEBase):
    def __init__(self,D_1=7e-3,D_2=5.33e-3,func_f=None,bc=None,
    range_grid=None, period=None, num_grid=None):
        super().__init__()
        self.D_1 = D_1
        self.D_2 = D_2
        self.range_grid = range_grid
        self.period = period
        self.num_grid = num_grid
        self.grid = pde.CartesianGrid(self.range_grid,self.num_grid,periodic=self.period)
        self.vx = pde.VectorField.from_expression(self.grid, ["((1-abs(x))**2+(1-abs(y)))/10","((1-abs(x))+(1-abs(y))**2)/10"])
        self._E = pde.VectorField.from_expression(self.grid, ["(2-abs(x))/2","(2-abs(y))/2"])
        if func_f is None:
            self.func_f = lambda u: - (self.vx * u).divergence(bc =self.bc_u)
        else: 
            self.func_f =  func_f
        self.bc_u = "periodic" if bc is None else bc

    #def div_func(self,u):
    #    u_grad = u.gradient(bc=self.bc_u)
    #    return u_grad.to_scalar(scalar=0) + u_grad.to_scalar(scalar=1)
        
    def func_E(self,u):
        E = 1 - u * self._E
        return (E * u).divergence(bc=self.bc_u)
    
    def evolution_rate(self,state,t=0):
        u = state[0]
        # div_u = self.div_func(u)
        return self.D_1 * u.laplace(bc=self.bc_u) + self.D_2 * self.func_E(u) + self.func_f(u)

# generate dataset
class NPModel(Dataset):
    def __init__(self, pde_model, path, curve_num=100, sample_size=100, 
    T_range=None, delta_t=1e-3, range_grid=None, period=True, num_grid=32,
     seed=None, initial=None, noise_sigma=0.1, dim=1):

        if T_range is None:
            T_range = [0,3]
        if range_grid is None:
            range_grid = [[-1,1],[-1,1]]
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
                self.sample_size+1,self.dim,self.num_grid[0],self.num_grid[1]))

            for j in range(self.sample_size+1):
                if j == 0:
                    c1 = np.random.uniform(1,2)
                    c2 = np.random.uniform(1,2)
                    c3 = np.random.uniform(0,2*np.pi)
                    #state =  pde.FieldCollection.scalar_random_uniform(self.dim,self.grid,vmin=self.initial[0]+0.3,vmax=self.initial[1]-0.3,rng=np.random.default_rng(self.seed))
                    state = pde.FieldCollection.from_scalar_expressions(self.grid,f"sin({c1}*(1-abs(x))+{c2}*(1-abs(y))+{c3})+1")
                    solution_collection[0,0,:,:] = state.data #+ np.random.normal(
                                   # scale=self.noise_sigma,size=(self.dim,self.num_grid[0],self.num_grid[1]))

                else:
                    state = self.pde_model.solve(
                        state,t_range=[0,self.observed_delta_t],dt = self.delta_t,tracker = None)

                    solution_collection[j,0,:,:] = state.data + np.random.normal(
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
def create_data(path_train = "./data/train_np",path_test = "./data/test_np",path_utest = "./data/utest_np",
    D_1=7e-3,D_2=5.33e-3,func_f=None,bc=None,
    batch_size=[64,16],train_curve_num=80,test_curve_num=20,sample_size=100,T_range=[0,3],
    delta_t=1e-3,range_grid=[-1,1],period=True,num_grid=32,
    seed_train=None,seed_test=None,seed_utest=None,initial=[0,1],noise_sigma=0.1,dim=1,rand=True):

    pde_model = NernstPlanckPDE(D_1=D_1,D_2 = D_2,func_f=func_f,bc=bc,range_grid=range_grid, period=period, num_grid=num_grid)

    dataset_train = NPModel(
        pde_model,path_train,curve_num=train_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_train,initial=initial,noise_sigma=noise_sigma,dim=dim)

    dataset_test = NPModel(
        pde_model,path_test,curve_num=test_curve_num,sample_size=sample_size,
        T_range=T_range,delta_t=delta_t,range_grid=range_grid,period=period,
        num_grid=num_grid,seed=seed_test,initial=initial,noise_sigma=noise_sigma,dim=dim)
    
    dataset_utest = NPModel(
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