# neural network approximate for f(u,w) and g(u,w)
import torch
import torch.nn as nn
import torch.nn.functional as F
import siren_pytorch as siren
from collections import OrderedDict
from torchdiffeq import odeint
from ignite.utils import convert_tensor

# parametric part
class ParametricPart(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self._dx = dx
        self._laplacian = nn.Parameter(torch.tensor(
                [ 1, -2,  1],
        ).float().view(1, 1, 3) / (self._dx * self._dx), requires_grad=False)

        self._gradient = nn.Parameter(torch.tensor(
                [ -1/2, 0,  1/2],
        ).float().view(1, 1, 3) / (self._dx), requires_grad=False)
        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.)),
            'D_3': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()

    def forward(self, state):
        U = state
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        self.params['D_3'] = self._params['D_3']
        U_ = F.pad(U, pad=(1,1), mode='circular')
        Delta_u = F.conv1d(U_, self._laplacian)
        grad_u = F.conv1d(U_, self._gradient)
        D_1 = list(self.params.values())[0]
        D_2 = list(self.params.values())[1]
        D_3 = list(self.params.values())[2]
        return D_1 * Delta_u + D_2 * grad_u  + D_3 * U
    
    def _map_forward(self, state):
        U = state
        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        U_ = F.pad(U, pad=(1,1), mode='circular')
        Delta_u = F.conv1d(U_, self._laplacian)
        grad_u = F.conv1d(U_, self._gradient)
        return {'delta_u':Delta_u, 'grad_u':grad_u, 'u': U}
        
    def get_value(self,x):
        batch_size, t_, u_w, h= x.shape
        x = x.contiguous()
        x = x.view(batch_size * t_, u_w, h)
        x = self._map_forward(x)
        delta_u = x['delta_u'].view(batch_size, t_, u_w, h)
        delta_u = delta_u.contiguous()
        grad_u = x['grad_u'].view(batch_size, t_, u_w,h)
        grad_u = grad_u.contiguous()
        u = x['u'].view(batch_size, t_, u_w,h)
        u = u.contiguous()
        return {'delta_u':delta_u, 'grad_u':grad_u, 'u': u}

# nonparametric part
class NonParametricModel(nn.Module):
    def __init__(self, state_dim=1, hidden=None,act=None,device='cpu'):
        super().__init__()
        if hidden is None:
            hidden = [16,64,64,16,1]
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
            self._initial_param()
            self.device=device
            
    def _initial_param(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
            
    def forward(self,x):
        dim_x = len(x.shape)

        # for solution u
        if dim_x == 3:
            batch_size, u_w, h = x.shape
            x = x.permute(0,2,1).contiguous()
            x = x.view(batch_size * h, u_w)
            x = self.net(x)
            x = x.view(batch_size, h, u_w)
            x = x.permute(0,2,1).contiguous()

        elif dim_x == 4:
            batch_size, t_, u_w, h = x.shape
            x = x.permute(0,1,3,2).contiguous()
            x = x.view(batch_size * t_ * h, u_w)
            x = self.net(x)
            x = x.view(batch_size, t_, h, u_w)
            x = x.permute(0,1,3,2).contiguous()

        return x

    def get_value(self,x):
        return self.forward(x)


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
            return self.para_model(state) + self.nonpara_model(state)

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
        working_solution = odeint(self._derivation,init_value,t,method='euler',options=dict(step_size=0.02))
        return working_solution.permute(1,0,2,3).contiguous()

    def _L2_norm(self,x):
        return torch.linalg.vector_norm(x)#,dim=(0,1,3,4))

    def _L2_inner_product(self,x,y):
        return abs(torch.sum(x*y))#,dim=(0,1,3,4))

    def _penalty(self,para_x,nonpara_x):
        u_para_x1 = para_x['delta_u']
        u_para_x2 = para_x['grad_u']
        u_para_x3 = para_x['u']
        u_nonpara_x = nonpara_x
        pen_u1 = self._L2_inner_product(u_para_x1,u_nonpara_x)/(self._L2_norm(u_para_x1))/(self._L2_norm(u_nonpara_x))
        pen_u2 = self._L2_inner_product(u_para_x2,u_nonpara_x)/(self._L2_norm(u_para_x2))/(self._L2_norm(u_nonpara_x))
        pen_u3 = self._L2_inner_product(u_para_x3,u_nonpara_x)/(self._L2_norm(u_para_x3))/(self._L2_norm(u_nonpara_x))
        return 1/(1-pen_u1) + 1/(1-pen_u2) + 1/(1-pen_u3) - 3

    def get_penalty(self,x):
        para_x = self.para_model.get_value(x)
        nonpara_x = self.nonpara_model.get_value(x)
        if self.penalty is None:
            return 0
        elif self.penalty == "orthogonal":
            return self._penalty(para_x,nonpara_x)
        else:  
            return self._L2_norm(nonpara_x)

    def para_value(self):
        return self.para_model.params


    


    
