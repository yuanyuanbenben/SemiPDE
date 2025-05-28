import torch
import torch.nn as nn
import torch.nn.functional as F
import siren_pytorch as siren
from collections import OrderedDict
from torchdiffeq import odeint
from ignite.utils import convert_tensor
import numpy as np

# one dimensional NS equation

# u_t + u*u_x - v*u_xx - F - f(t,x) = 0
# F = -(1/2*C_D*h_v*b_v*u*|u| + pi/4*C_M*h_v*b_v^2u_t)

# (1+ pi/4*C_M*h_v*b_v^2) u_t + u * u_x - v*u_xx + 1/2*C_D*h_v*b_v*u*|u| - f(t,x) = 0
# initial condition u(0,x) known
# boundary condition u_x(0,x) known

# parametric part
class ParametricPart(nn.Module):
    def __init__(self,h_v,b_v,dx=0.2):
        super().__init__()
        self.h_v = h_v
        self.b_v = b_v
        self._dx = dx
        self._laplacian = nn.Parameter(torch.tensor(
                [ 1,-2,  1],
        ).float().view(1, 1, 3) / (self._dx*self._dx), requires_grad=False)
        self._gradient = nn.Parameter(torch.tensor(
                [ -1/2, 0,  1/2],
        ).float().view(1, 1, 3) / (self._dx), requires_grad=False)
        self._params = nn.ParameterDict({
            'C_D': nn.Parameter(torch.tensor(0.)), 
            'nv': nn.Parameter(torch.tensor(0.))
        })
        self.params = OrderedDict()

    def forward(self, state):
        self.params['C_D'] = self._params['C_D']
        self.params['nv'] = self._params['nv']
        C_D, nv = list(self.params.values())
        U = state
        batch_size, d = U.shape
        U = U.contiguous()
        U = U.view(batch_size,1,d)
        U_ = F.pad(U, pad=(1,1), mode='replicate')
        u_xx = F.conv1d(U_, self._laplacian)
        u_x = F.conv1d(U_, self._gradient)
        u_xx = u_xx.view(batch_size,d)
        u_x = u_x.view(batch_size,d)
        u_xx.contiguous()
        u_x.contiguous()
        u_xx[:,0] = u_xx[:,1]
        u_xx[:,-1] = u_xx[:,-2]
        u_x[:,0] = u_x[:,1]
        u_x[:,-1] = u_x[:,-2]
        return - C_D * 0.5 * self.h_v * self.b_v * state * torch.abs(state) + nv * u_xx - state * u_x 

# nonparametric part
class NonParametricModel(nn.Module):
    def __init__(self, state_dim=2,hidden=None,act=None):
        super().__init__()
        if hidden is None:
            hidden = [16,64,64,16,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear" + 'net', nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act" + 'net', nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act" + 'net', nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act" + 'net', siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear"+ 'net',
                                nn.Linear(hidden[i], hidden[i+1],bias=True))
        # self.net2 = nn.Sequential()
        # self.net2.add_module(str(0) + "linear" + 'net2', nn.Linear(state_dim, hidden[0],bias=True))
        # for i in range(self.layer_num - 1):
        #     if act == 'ReLU':
        #         self.net2.add_module(str(i) + "Act" + 'net2', nn.ReLU())
        #     if act == 'Tanh':
        #         self.net2.add_module(str(i) + "Act" + 'net2', nn.Tanh())
        #     if act == 'Sine':
        #         self.net2.add_module(str(i) + "Act" + 'net2', siren.Sine())
        #     #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
        #     self.net2.add_module(str(i+1) + "linear"+ 'net2',
        #                           nn.Linear(hidden[i], hidden[i+1],bias=True))
        self._random_seed(20230914)
        self._initial_param()
        
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
        # for name, param in self.net2.named_parameters():
        #     if name.endswith('weight'):
        #         nn.init.xavier_normal_(param)
        #     elif name.endswith('bias'):
        #         nn.init.zeros_(param)
                
    def change_x_position(self,x_position):
        self.batch,self.x = x_position.size()
        self.x_position = x_position.view(self.batch*self.x,1).contiguous()
        
    def _random_seed(self,seed):
        torch.manual_seed(seed)
    
    def forward(self,t):
        t_ = torch.zeros_like(self.x_position)+t
        input = torch.cat((t_,self.x_position),dim=1)
        f = self.net(input)
        f = f.view(self.batch,self.x).contiguous()
        return f
        # u_z
        # U = state
        # batch_size, h = U.shape
        # U = U.contiguous()
        # U = U.view(batch_size,1,h)
        # U_ = F.pad(U, pad=(1,1), mode='replicate')
        # u_z = F.conv1d(U_, self._gradient)
        # u_z = u_z.view(batch_size,h)
        # u_z.contiguous()
        # # f
        # f = state.contiguous()
        # f = f.view(batch_size * h, 1)
        # f = self.net(f)
        # f = f.view(batch_size, h)
        # f = f.contiguous()
        # # g
        # g = state.contiguous()
        # g = g.view(batch_size * h, 1)
        # g = self.net2(g)
        # g = g.view(batch_size, h)
        # g = g.contiguous()
        # # f * u_z
        # U_ = f * u_z
        # # (f * u_z)_z
        # U_ = U_.contiguous()
        # U_ = U_.view(batch_size,1,h)
        # U_ = F.pad(U_,pad=(1,1),mode='replicate')
        # Delta_u = F.conv1d(U_, self._gradient)
        # Delta_u = Delta_u.view(batch_size,h)
        # Delta_u.contiguous()
        
        # return Delta_u *0.00001 + g*0.001
        # return g * 0.1
    
    def _loss(self,input_):
        g = self.net(input_)
        return torch.mean(torch.square(g)) * 1e-3

# derivation combine 
class DerivationCombineModel(nn.Module):
    def __init__(self,para_part,unknow_part,h_v,b_v,mode=None):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self.mode = mode
        self._multi = 1 / (1+torch.pi/2*h_v*b_v**2)
        
    def forward(self,t,state):
        if self.mode is None:
            return self.para_model(state) * self._multi
        else:
            return (self.para_model(state) + self.nonpara_model(t)*0.01) * self._multi 
        
# the whole model 
class SemiParametericModel(nn.Module):
    def __init__(self,para_part,unknow_part,h_v,b_v,mode=None,penalty=None):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self.mode = mode
        self.penalty = penalty
        self._derivation = DerivationCombineModel(self.para_model,self.nonpara_model,h_v,b_v,self.mode)

    def forward(self, init_value, t):
        working_solution = odeint(self._derivation,init_value,t,method='euler',options=dict(step_size=0.02))
        return working_solution.permute(1,0,2).contiguous()

    def para_value(self):
        return self.para_model.params


    
