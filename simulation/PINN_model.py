import torch as torch
import torch.nn as nn
import deepxde as dde
import siren_pytorch as siren
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchdiffeq import odeint
from ignite.utils import convert_tensor
import gc

class PINN_Net(nn.Module):
    def __init__(self, state_dim=1, hidden=None,act=None):
        super().__init__()
        if hidden is None:
            hidden = [32,128,128,32,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",
                                nn.Linear(hidden[i], hidden[i+1],bias=True))
            self._random_seed(20230914)
            self._initial_param()
            
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        

    def forward(self, grid_point):
        return self.net(grid_point)


class NonParaGCDEquation(nn.Module):
    def __init__(self, state_dim=1, hidden=None,lamda = 0,act=None, device='cpu'):
        super().__init__()
            
        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.)),
            'D_3': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()
        self.myLoss = nn.MSELoss()
        self.lamda = lamda
        
        if hidden is None:
            hidden = [32,128,128,32,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",
                                nn.Linear(hidden[i], hidden[i+1],bias=True))
        self._random_seed(20230914)
        self._initial_param()
            
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
    def forward(self,position,solution_net):
        # ReactionConvectionDiffusionPDE
        # u_t = D_1 u_xx + D_2 u_x + D_3 u + f(u)
        # D_1 = 1 D_2 = D_3 = 0.1 
        # u(t,0) = 0, u(0,x) = (1+0.1(20-x)^2)^{-1}
        # t [1,20],  x [1,40]
        
        # position: batch size * state ->(t,x,y)
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        self.params['D_3'] = self._params['D_3']
        (D_1, D_2, D_3) = list(self.params.values())
        U_ =  solution_net(position)
        F_ = self.net(U_)
        U_later = solution_net(position + torch.tensor([1e-1,0]))
        u_t = (U_later - U_)
        U_x_later = solution_net(position + torch.tensor([0,0.1]))
        U_x_before = solution_net(position + torch.tensor([0,-0.1]))
        # U_x_later2 = solution_net(position + torch.tensor([0,0.05]))
        # U_x_before2 = solution_net(position + torch.tensor([0,-0.05]))
        grad_u = (U_x_later - U_x_before)/2
        lap_u = (U_x_later - U_ *2 + U_x_before)
        # grad_u = (U_x_later2 - U_x_before2)/1
        # lap_u = (U_x_later - U_ *2 + U_x_before)/1
        
        loss_eq = self.myLoss(u_t, D_1 * lap_u * 10 + D_2 * grad_u + D_3 * U_ + F_*0.01) + self.lamda * torch.mean(torch.square(F_))
        
        return {'loss_eq': loss_eq}
    
    

class ParaGCDEquation(nn.Module):
    def __init__(self):
        super().__init__()
        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.0)),
            'D_3': nn.Parameter(torch.tensor(0.0)),
        })

        self.params = OrderedDict()
        self.myLoss = nn.MSELoss()
        
    def init_func(self,init_position):
        return 1/(1+0.1*(20-init_position[:,1:2])**2)
    
    
    def forward(self,position,solution_net):
        # ReactionConvectionDiffusionPDE
        # u_t = D_1 u_xx + D_2 u_x + D_3 u
        # D_1 = 1 D_2 = D_3 = 0.1 
        # u(t,0) = 0, u(0,x) = (1+0.1(20-x)^2)^{-1}
        # t [1,20],  x [1,40]
        
        # position: batch size * state ->(t,x,y)
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        self.params['D_3'] = self._params['D_3']
        (D_1, D_2, D_3) = list(self.params.values())
        # position.requires_grad_(True)
        # with torch.no_grad():
        U_ =  solution_net(position)
        U_later = solution_net(position + torch.tensor([1e-1,0]))
        u_t = (U_later - U_)
        U_x_later = solution_net(position + torch.tensor([0,0.1]))
        U_x_before = solution_net(position + torch.tensor([0,-0.1]))
        # U_x_later2 = solution_net(position + torch.tensor([0,0.05]))
        # U_x_before2 = solution_net(position + torch.tensor([0,-0.05]))
        grad_u = (U_x_later - U_x_before)/2
        lap_u = (U_x_later - U_ *2 + U_x_before)
        # u_t = dde.grad.jacobian(U_,position,i=0,j=0) / 10
        # grad_u = dde.grad.jacobian(U_,position,i=0,j=1) / 10
        # lap_u = dde.grad.hessian(U_,position,i=1,j=1) / 100
        
        # init = solution_net(init_position)
        # true_init = self.init_func(init_position)
        
        loss_eq = self.myLoss(u_t, D_1 * lap_u *10 + D_2 * grad_u  + D_3 * U_) #+ 10 * self.myLoss(init,true_init)
        return {'loss_eq': loss_eq}
    
    
class GFNEquation(nn.Module):
    def __init__(self, state_dim=1, hidden=None,act=None):
        super().__init__()
        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()
        self.myLoss = nn.MSELoss()
        # self.lamda = lamda
        
        if hidden is None:
            hidden = [32,128,128,32,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",nn.Linear(hidden[i], hidden[i+1],bias=True))
        self._random_seed(20230914)
        self._initial_param()
            
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
    def forward(self,position,solution_net,para=False):
        # GFN_PDE
        # u_t = D_1 Laplace u + f(u,w), 
        # w_t = D_2 Laplace w + g(u,w), 
        
        
        # position: batch size * state ->(t,x,y)
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        self.D_1, self.D_2 = list(self.params.values())
        self.U_ = solution_net(position)
        if para:
            self.F_ = self.net(self.U_) * 0.0
        else:
            self.F_ = self.net(self.U_) * 1e-2
        self.u_t = dde.grad.jacobian(self.U_,position,i=0,j=0)
        self.w_t = dde.grad.jacobian(self.U_,position,i=1,j=0)
        self.lap_u = dde.grad.hessian(self.U_,position,component=0,i=1,j=1) + dde.grad.hessian(self.U_,position,component=0,i=2,j=2)
        self.lap_w = dde.grad.hessian(self.U_,position,component=1,i=1,j=1) + dde.grad.hessian(self.U_,position,component=1,i=2,j=2)
        # U_later = solution_net(position + torch.tensor([1e-1,0]))
        # u_t = (U_later - U_)
        # U_x_later = solution_net(position + torch.tensor([0,0.1]))
        # U_x_before = solution_net(position + torch.tensor([0,-0.1]))
        # # U_x_later2 = solution_net(position + torch.tensor([0,0.05]))
        # # U_x_before2 = solution_net(position + torch.tensor([0,-0.05]))
        # grad_u = (U_x_later - U_x_before)/2
        # lap_u = (U_x_later - U_ *2 + U_x_before)
        # grad_u = (U_x_later2 - U_x_before2)/1
        # lap_u = (U_x_later - U_ *2 + U_x_before)/1
        
        self.loss_eq = self.myLoss(self.u_t, self.D_1* self.lap_u + self.F_[:,0:1]) + self.myLoss(self.w_t, self.D_2* self.lap_w + self.F_[:,1:2])
        
class NSEquation(nn.Module):
    def __init__(self, state_dim=1, hidden=None,act=None):
        super().__init__()
        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor([0.,0.,0.])), 
            'D_2': nn.Parameter(torch.tensor([0.,0.,0.])),
        })

        self.params = OrderedDict()
        self.myLoss = nn.MSELoss()
        # self.lamda = lamda
        
        if hidden is None:
            hidden = [32,128,128,32,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        # f(x)
        self.net1 = nn.Sequential()
        self.net1.add_module(str(0) + "linear", nn.Linear(state_dim[0], hidden[0],bias=True))
        for i in range(self.layer_num):
            if act == 'ReLU':
                self.net1.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net1.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net1.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            if i < (self.layer_num-1):
                self.net1.add_module(str(i+1) + "linear",nn.Linear(hidden[i], hidden[i+1],bias=True))
            else: 
                self.net1.add_module(str(i+1) + "linear",nn.Linear(hidden[i], 3,bias=True))
        # g(x)
        self.net2 = nn.Sequential()
        self.net2.add_module(str(0) + "linear", nn.Linear(state_dim[1], hidden[0],bias=True))
        for i in range(self.layer_num):
            if act == 'ReLU':
                self.net2.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net2.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net2.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            if i < (self.layer_num-1):
                self.net2.add_module(str(i+1) + "linear",nn.Linear(hidden[i], hidden[i+1],bias=True))
            else: 
                self.net2.add_module(str(i+1) + "linear",nn.Linear(hidden[i], 1,bias=True))
            
        self._random_seed(20230914)
        self._initial_param()
            
    def _initial_param(self):
        for name, param in self.net1.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
        for name, param in self.net2.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
    def forward(self,position,solution_net):
        # GFN_PDE
        # u_t = D_1 Laplace u + f(u,w), 
        # w_t = D_2 Laplace w + g(u,w), 
        
        
        # position: batch size * state ->(t,x,y)
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        self.D_1, self.D_2 = list(self.params.values())
        self.U_ = solution_net(position)
        self.F_1 = self.net1(position[:,1:4]) * 1e-2
        self.F_2 = self.net2(torch.cat((position[:,1:4],self.U_),dim=1)) * 1e-2
        self.u_div = dde.grad.jacobian(self.U_,position,i=0,j=1)+ dde.grad.jacobian(self.U_,position,i=1,j=2) + dde.grad.jacobian(self.U_,position,i=2,j=3)
        self.F_2_u_x = self.F_2 * self.u_div
        self.G_ = torch.cat((dde.grad.jacobian(self.F_2_u_x,position,i=0,j=1), 
                            dde.grad.jacobian(self.F_2_u_x,position,i=0,j=2),
                            dde.grad.jacobian(self.F_2_u_x,position,i=0,j=3)),dim=1)
        
        self.u_t = torch.cat((dde.grad.jacobian(self.U_,position,i=0,j=0),
                            dde.grad.jacobian(self.U_,position,i=1,j=0),
                            dde.grad.jacobian(self.U_,position,i=2,j=0)),dim=1)
        
        self.u_grad_u_x = dde.grad.jacobian(self.U_,position,i=0,j=1) * self.U_[:,0:1] + dde.grad.jacobian(self.U_,position,i=1,j=1) * self.U_[:,1:2] + dde.grad.jacobian(self.U_,position,i=2,j=1) * self.U_[:,2:3]
        self.u_grad_u_y = dde.grad.jacobian(self.U_,position,i=0,j=2) * self.U_[:,0:1] + dde.grad.jacobian(self.U_,position,i=1,j=2) * self.U_[:,1:2] + dde.grad.jacobian(self.U_,position,i=2,j=2) * self.U_[:,2:3]
        self.u_grad_u_z = dde.grad.jacobian(self.U_,position,i=0,j=3) * self.U_[:,0:1] + dde.grad.jacobian(self.U_,position,i=1,j=3) * self.U_[:,1:2] + dde.grad.jacobian(self.U_,position,i=2,j=3) * self.U_[:,2:3]
        self.u_grad_u = torch.cat((self.u_grad_u_x, self.u_grad_u_y, self.u_grad_u_z),dim=1)
        
        self.lap_u_x = dde.grad.hessian(self.U_,position,component=0,i=1,j=1) + dde.grad.hessian(self.U_,position,component=0,i=2,j=2) + dde.grad.hessian(self.U_,position,component=0,i=3,j=3)
        self.lap_u_y = dde.grad.hessian(self.U_,position,component=1,i=1,j=1) + dde.grad.hessian(self.U_,position,component=1,i=2,j=2) + dde.grad.hessian(self.U_,position,component=1,i=3,j=3)
        self.lap_u_z = dde.grad.hessian(self.U_,position,component=2,i=1,j=1) + dde.grad.hessian(self.U_,position,component=2,i=2,j=2) + dde.grad.hessian(self.U_,position,component=2,i=3,j=3)
        self.lap_u = torch.cat((self.lap_u_x,self.lap_u_y,self.lap_u_z),dim=1)
        self.grad_u_div = torch.cat((dde.grad.jacobian(self.u_div,position,i=0,j=1), 
                                    dde.grad.jacobian(self.u_div,position,i=0,j=2),
                                    dde.grad.jacobian(self.u_div,position,i=0,j=3)),dim=1)  
        self.p_ = torch.sum(self.U_ * (1-torch.abs(position[:,1:4])),dim=1,keepdim=True)
        self.p_grad = torch.cat((dde.grad.jacobian(self.p_,position,i=0,j=1),
                                dde.grad.jacobian(self.p_,position,i=0,j=2),
                                dde.grad.jacobian(self.p_,position,i=0,j=3)),dim=1)
        
        self.loss_eq = self.myLoss(self.u_t,-1e-3*self.u_grad_u + self.D_1 * (self.lap_u - self.grad_u_div/3) - self.D_2 * self.p_grad + self.F_1 + self.G_)
        
    def cache_out(self):
        del self.U_, self.F_1, self.F_2, self.u_div, self.F_2_u_x, self.G_, self.u_t, self.u_grad_u_x, self.u_grad_u_y, self.u_grad_u_z
        del self.u_grad_u, self.lap_u_x, self.lap_u_y, self.lap_u_z, self.lap_u, self.grad_u_div, self.p_, self.p_grad, self.loss_eq
        gc.collect()
        
class NPEquation(nn.Module):
    def __init__(self, state_dim=2, hidden=None,act=None):
        super().__init__()
        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()
        self.myLoss = nn.MSELoss()
        # self.lamda = lamda
        
        if hidden is None:
            hidden = [32,128,128,32,2]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",nn.Linear(hidden[i], hidden[i+1],bias=True))
        self._random_seed(20230914)
        self._initial_param()
            
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
    def forward(self,position,solution_net):
        
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        self.D_1, self.D_2 = list(self.params.values())
        self.U_ = solution_net(position)
        self.F_ = self.net(position[:,1:3]) * 1e-2
        self.fu = self.F_ * self.U_
        self.fu_div = dde.grad.jacobian(self.fu,position,i=0,j=1) + dde.grad.jacobian(self.fu,position,i=1,j=2)
        
        self.u_t = dde.grad.jacobian(self.U_,position,i=0,j=0)
        self.lap_u = dde.grad.hessian(self.U_,position,component=0,i=1,j=1) + dde.grad.hessian(self.U_,position,component=0,i=2,j=2)
        self.E_ = (1 - (2 - torch.abs(position[:,1:3])) * self.U_)/2
        self.E_U = self.E_ * self.U_
        self.E_U_div = dde.grad.jacobian(self.E_U,position,i=0,j=1) + dde.grad.jacobian(self.E_U,position,i=1,j=2)
        
        self.loss_eq = self.myLoss(self.u_t, self.D_1* self.lap_u + self.D_2 * self.E_U_div - self.fu_div)
        
class RDEquation(nn.Module):
    def __init__(self, state_dim=1, hidden=None,act=None):
        super().__init__()
        self._params = nn.ParameterDict({
            'D': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()
        self.myLoss = nn.MSELoss()
        # self.lamda = lamda
        
        if hidden is None:
            hidden = [32,128,128,32,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act" , nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act" , nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",nn.Linear(hidden[i], hidden[i+1],bias=True))
        self._random_seed(20230914)
        self._initial_param()
            
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
    def forward(self,position,solution_net,para=False):
        # GFN_PDE
        # u_t = D_1 Laplace u + f(u,w), 
        # w_t = D_2 Laplace w + g(u,w), 
        
        
        # position: batch size * state ->(t,x,y)
        self.params['D'] = self._params['D']
        self.D = list(self.params.values())[0]
        self.U_ = solution_net(position)
        self.F_ = self.net(self.U_) * 1e-2
        if para:
            self.F_ = self.F_ * 0.0
        
        self.u_t = dde.grad.jacobian(self.U_,position,i=0,j=0)
        self.lap_u = dde.grad.hessian(self.U_,position,component=0,i=1,j=1)
        
        self.loss_eq = self.myLoss(self.u_t, self.D * self.lap_u + self.F_)
        