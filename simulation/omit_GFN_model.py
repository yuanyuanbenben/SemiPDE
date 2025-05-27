# neural network approximate for f(u,w) and g(u,w)
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchdiffeq import odeint

# parametric part
class ParametricPart(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self._dx = dx
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx * self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.)),
        })

        self.params = OrderedDict()

    def forward(self, state):
        U = state[:,:1,:,:]
        W = state[:,1:,:,:]

        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
                
        U_ = F.pad(U, pad=(1,1,1,1), mode='circular')
        Delta_u = F.conv2d(U_, self._laplacian)
        
        W_ = F.pad(W, pad=(1,1,1,1), mode='circular')
        Delta_w = F.conv2d(W_, self._laplacian)

        (D_1, D_2) = list(self.params.values())
        u_t = D_1 * Delta_u 
        w_t = D_2 * Delta_w

        return torch.cat([u_t,w_t], dim=1)
    
    def _map_forward(self, state):
        U = state[:,:1,:,:]
        W = state[:,1:,:,:]

        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
                
        U_ = F.pad(U, pad=(1,1,1,1), mode='circular')
        u_t= F.conv2d(U_, self._laplacian)
        
        W_ = F.pad(W, pad=(1,1,1,1), mode='circular')
        w_t = F.conv2d(W_, self._laplacian)

        return torch.cat([u_t,w_t], dim=1)
        
    def get_value(self,x):
        batch_size, t_, u_w, h, w = x.shape
        x = x.contiguous()
        x = x.view(batch_size * t_, u_w, h, w)
        x = self._map_forward(x)
        x = x.view(batch_size, t_, u_w, h, w)
        x = x.contiguous()
        return x

# nonparametric part
class NonParametricModel(nn.Module):
    def __init__(self,state_dim=2,hidden=[16,64,64,16]):
        super().__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(self.state_dim,self.hidden[0]),
            nn.ReLU(),
            nn.Linear(self.hidden[0],self.hidden[1]),
            nn.ReLU(),
            nn.Linear(self.hidden[1],self.hidden[2]),
            nn.ReLU(),
            nn.Linear(self.hidden[2],self.hidden[3]),
            nn.ReLU(),
            nn.Linear(self.hidden[3],self.state_dim)
        )

    def forward(self,x):
        dim_x = len(x.shape)

        # for solution u
        if dim_x == 4:
            batch_size, u_w, h, w = x.shape
            x = x.permute(0,2,3,1).contiguous()
            x = x.view(batch_size * h * w, u_w)
            x = self.net(x)
            x = x.view(batch_size, h, w, u_w)
            x = x.permute(0,3,1,2).contiguous()

        # for get value for data grid
        if dim_x == 5:
            batch_size, t_, u_w, h, w = x.shape
            x = x.permute(0,1,3,4,2).contiguous()
            x = x.view(batch_size * t_ * h * w, u_w)
            x = self.net(x)
            x = x.view(batch_size, t_, h, w, u_w)
            x = x.permute(0,1,4,2,3).contiguous()

        return x

    def get_value(self,x):
        return self.forward(x)

# derivation combine 
class DerivationCombineModel(nn.Module):
    def __init__(self,para_part,unknow_part):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part

    def forward(self,t,state):
        return self.para_model(state) + self.nonpara_model(state)

# the whole model 
class SemiParametericModel(nn.Module):
    def __init__(self,para_part,unknow_part):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self._derivation = DerivationCombineModel(self.para_model,self.nonpara_model)

    def forward(self, init_value, t):
        working_solution = odeint(self._derivation,init_value,t,method="rk4")
        return working_solution.permute(1,0,2,3,4).contiguous()

    def get_penalty(self,x):
        para_x = self.para_model.get_value(x)
        nonpara_x = self.nonpara_model.get_value(x)
        #penalty = self._L2_inner_product(para_x,nonpara_x)/(self._L2_norm(para_x)+1e-5)/(self._L2_norm(nonpara_x)+1e-5)
        return self._penalty(para_x,nonpara_x)

    def para_value(self):
        return self.para_model.params


    
