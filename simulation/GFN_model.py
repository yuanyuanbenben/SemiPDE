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
    def __init__(self, state_dim=2, hidden=None, act=None, device='cpu'):
        super().__init__()
        if hidden is None:
            hidden = [16,64,16,2]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act", nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act", nn.Tanh())
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
        if dim_x == 4:
            batch_size, u_w, h, w = x.shape
            x = x.permute(0,2,3,1).contiguous()
            x = x.view(batch_size * h * w, u_w)
            x = self.net(x)
            x = x.view(batch_size, h, w, u_w)
            x = x.permute(0,3,1,2).contiguous()

        elif dim_x == 5:
            batch_size, t_, u_w, h, w = x.shape
            x = x.permute(0,1,3,4,2).contiguous()
            x = x.view(batch_size * t_ * h * w, u_w)
            x = self.net(x)
            x = x.view(batch_size, t_, h, w, u_w)
            x = x.permute(0,1,4,2,3).contiguous()

        return x

    def get_value(self,x):
        return self.forward(x)
    
    def get_grad(self,x):
        x.requires_grad_(True)
        x_value = self.get_value(x)
        x_grad = torch.autograd.grad(x_value,x,convert_tensor(torch.ones(x_value.size()),self.device),retain_graph=True,create_graph=True)
        x.requires_grad_(False)
        return x_grad[0]

# derivation combine 
class DerivationCombineModel(nn.Module):
    def __init__(self,para_part,unknow_part,mode=None):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self.mode=mode

    def forward(self,t,state):
        if self.mode is None:
            return self.para_model(state)
        else:
            return self.para_model(state) + self.nonpara_model(state)

# the whole model 
class SemiParametericModel(nn.Module):
    def __init__(self,para_part,unknow_part,mode=None,penalty=None,device='cpu'):
        super().__init__()
        self.para_model = para_part
        self.nonpara_model = unknow_part
        self.mode = mode
        self.penalty = penalty
        self._derivation = DerivationCombineModel(self.para_model,self.nonpara_model,self.mode)
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, init_value, t):
        working_solution = odeint(self._derivation,init_value,t,method='euler',options=dict(step_size=0.02))
        return working_solution.permute(1,0,2,3,4).contiguous()
    
    def _L2_norm(self,x):
        return torch.linalg.vector_norm(x)#,dim=(0,1,3,4))

    def _L2_inner_product(self,x,y):
        return abs(torch.sum(x*y))#,dim=(0,1,3,4))
    
    def _penalty(self,para_x,nonpara_x):
        u_para_x = para_x[:,:,0,:,:]
        w_para_x = para_x[:,:,1,:,:]
        u_nonpara_x = nonpara_x[:,:,0,:,:]
        w_nonpara_x = nonpara_x[:,:,1,:,:]
        pen_u = self._L2_inner_product(u_para_x,u_nonpara_x)/(self._L2_norm(u_para_x)+1e-5)/(self._L2_norm(u_nonpara_x)+ 1e-5)
        pen_w = self._L2_inner_product(w_para_x,w_nonpara_x)/(self._L2_norm(w_para_x)+1e-5)/(self._L2_norm(w_nonpara_x)+ 1e-5)
        return 1/(1-pen_u) + 1/(1-pen_w) - 2
    
    def get_penalty(self,x):
        para_x = self.para_model.get_value(x)
        nonpara_x = self.nonpara_model.get_value(x)
        if self.penalty is None:
            return 0
        elif self.penalty == "orthogonal":
            return self._penalty(para_x,nonpara_x)
        elif self.penalty == 'sobolev':
            grad = self.nonpara_model.get_grad(x)
            zero_grad = torch.zeros(grad.size())
            zero_grad = convert_tensor(zero_grad,self.device)
            return self.loss(grad,zero_grad) + self.loss(nonpara_x,zero_grad)
        else:  
            return self._L2_norm(nonpara_x)

    def para_value(self):
        return self.para_model.params


    


class NonparaModel(nn.Module):
    def __init__(self, state_dim=3, hidden=None,act=None):
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
        return self.net(x)

    def get_value(self,x):
        return self.forward(x)
