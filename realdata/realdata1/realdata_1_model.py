import torch
import torch.nn as nn
import torch.nn.functional as F
import siren_pytorch as siren
from collections import OrderedDict
from torchdiffeq import odeint
from ignite.utils import convert_tensor

# model
# u_t = D nabla u + f(u)
# boundary condition u_x = 0
# initial condition u(x,0) known

# parametric part
class ParametricPart(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self._dx = dx
        self._laplacian = nn.Parameter(torch.tensor(
                [ 1, -2,  1],
        ).float().view(1, 1, 3) / (self._dx * self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D': nn.Parameter(torch.tensor(0.)), 
        })

        self.params = OrderedDict()

    def forward(self, state):
        U = state
        batch_size, h= U.shape
        U = U.contiguous()
        U = U.view(batch_size,1,h)
        self.params['D'] = self._params['D']
        U_ = F.pad(U, pad=(1,1), mode='replicate')
        Delta_u = F.conv1d(U_, self._laplacian)
        
        Delta_u = Delta_u.view(batch_size,h)
        Delta_u.contiguous()
        D = list(self.params.values())[0]
        return D * Delta_u * 1000
    
class ModifyParametricPart(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self._dx = dx
        self._gradient = nn.Parameter(torch.tensor(
                [ -1, 0,  1],
        ).float().view(1, 1, 3) / (self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D': nn.Parameter(torch.tensor(0.)), 
        })
        self.params = OrderedDict()

    
    def forward(self,state):
        # apply D * partial_x ((C/K)partial_x C)
        U = state
        batch_size, h = U.shape
        U = U.contiguous()
        U = U.view(batch_size,1,h)
        self.params['D'] = self._params['D']
        U_ = F.pad(U, pad=(1,1), mode='replicate')
        Delta_u = F.conv1d(U_, self._gradient)
        Delta_u = Delta_u.view(batch_size,h)
        Delta_u.contiguous()
        
        U_ = Delta_u * state / 1.7e-3
        U_ = U_.contiguous()
        U_ = U_.view(batch_size,1,h)
        U_ = F.pad(U_,pad=(1,1),mode='replicate')
        Delta_u = F.conv1d(U_, self._gradient)
        Delta_u = Delta_u.view(batch_size,h)
        Delta_u.contiguous()
        
        D = list(self.params.values())[0]
        return D * Delta_u * 1000
    
# parametric part
class TrueParametricPart(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self._dx = dx
        self._laplacian = nn.Parameter(torch.tensor(
                [ 1, -2,  1],
        ).float().view(1, 1, 3) / (self._dx * self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D': nn.Parameter(torch.tensor(0.)), 
            'lamda': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()

    def forward(self, state):
        U = state
        batch_size, h= U.shape
        U = U.contiguous()
        U = U.view(batch_size,1,h)
        self.params['D'] = self._params['D']
        self.params['lamda'] = self._params['lamda']
        U_ = F.pad(U, pad=(1,1), mode='replicate')
        Delta_u = F.conv1d(U_, self._laplacian)
        
        Delta_u = Delta_u.view(batch_size,h)
        Delta_u.contiguous()
        D = list(self.params.values())[0]
        lamda = list(self.params.values())[1]
        return D * Delta_u * 1000 + lamda * state * (1- state/1.7e-3)
    
    
    
class TrueModifyParametricPart(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self._dx = dx
        self._gradient = nn.Parameter(torch.tensor(
                [ -1, 0,  1],
        ).float().view(1, 1, 3) / (self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D': nn.Parameter(torch.tensor(0.)), 
            'lamda': nn.Parameter(torch.tensor(0.)),
        })
        self.params = OrderedDict()

    
    def forward(self,state):
        # apply D * partial_x ((C/K)partial_x C)
        U = state
        batch_size, h = U.shape
        U = U.contiguous()
        U = U.view(batch_size,1,h)
        self.params['D'] = self._params['D']
        self.params['lamda'] = self._params['lamda']
        U_ = F.pad(U, pad=(1,1), mode='replicate')
        Delta_u = F.conv1d(U_, self._gradient)
        Delta_u = Delta_u.view(batch_size,h)
        Delta_u.contiguous()
        
        U_ = Delta_u * state / 1.7e-3
        U_ = U_.contiguous()
        U_ = U_.view(batch_size,1,h)
        U_ = F.pad(U_,pad=(1,1),mode='replicate')
        Delta_u = F.conv1d(U_, self._gradient)
        Delta_u = Delta_u.view(batch_size,h)
        Delta_u.contiguous()
        
        D = list(self.params.values())[0]
        lamda = list(self.params.values())[1]
        return D * Delta_u * 1000  +  lamda * state * (1- state/1.7e-3)
    
# nonparametric part
class NonParametricModel(nn.Module):
    def __init__(self, state_dim=1, hidden=None,act=None):
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
        self._initial_param()
        
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
    def forward(self,x):
        # for solution u
        x = x.contiguous()
        batch_size, h = x.shape
        x = x.view(batch_size * h, 1)
        x = self.net(x)
        x = x.view(batch_size, h)
        x = x.contiguous()
        return x

# derivation combine 
class DerivationCombineModel(nn.Module):
    def __init__(self,para_part,unknow_part,mode=None):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self.mode = mode

    def forward(self,t,state):
        if self.mode is None:
            return self.para_model(state)
        else:
            return self.para_model(state) + self.nonpara_model(state * 1000 - 1) * 0.00001

# the whole model 
class SemiParametericModel(nn.Module):
    def __init__(self,para_part,unknow_part,mode=None,penalty=None):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self.mode = mode
        self.penalty = penalty
        self._derivation = DerivationCombineModel(self.para_model,self.nonpara_model,self.mode)

    def forward(self, init_value, t):
        working_solution = odeint(self._derivation,init_value,t,method='euler',options=dict(step_size=1))
        return working_solution.permute(1,0,2).contiguous()

    def para_value(self):
        return self.para_model.params


    
