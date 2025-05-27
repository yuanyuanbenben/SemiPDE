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
    def __init__(self, dx, device='cpu',tx=32):
        super().__init__()
        self._dx = dx
        self.device = device
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx * self._dx), requires_grad=False)
        self._tx=tx
        
        
        self._grad_x = nn.Parameter(torch.tensor(
            [
                [ 0,  -1/2,  0],
                [ 0, 0,  0],
                [ 0,  1/2,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx), requires_grad=False)

        self._grad_y= nn.Parameter(torch.tensor(
            [
                [ 0,  0,  0],
                [ -1/2, 0,  1/2],
                [ 0,  0,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor(0.)), 
            'D_2': nn.Parameter(torch.tensor(0.)),
        })
        # axis_1dx = nn.Parameter((2-torch.abs(torch.tensor(range(64)).float()/32-63/64))/2,requires_grad=False)
        # axis_1dy = nn.Parameter((2-torch.abs(torch.tensor(range(64)).float()/32-63/64))/2,requires_grad=False)
        # self.axis_x = convert_tensor(axis_1dx.view(1, 1, 64, 1),device=self.device)
        # self.axis_y = convert_tensor(axis_1dy.view(1, 1, 1, 64),device=self.device)
        axis_1dx = nn.Parameter((2-torch.abs(torch.tensor(range(self._tx)).float()/self._tx*2-(self._tx-1)/self._tx))/2,requires_grad=False)
        axis_1dy = nn.Parameter((2-torch.abs(torch.tensor(range(self._tx)).float()/self._tx*2-(self._tx-1)/self._tx))/2,requires_grad=False)
        self.axis_x = convert_tensor(axis_1dx.view(1, 1, self._tx, 1),device=self.device)
        self.axis_y = convert_tensor(axis_1dy.view(1, 1, 1, self._tx),device=self.device)
        self.params = OrderedDict()

    def forward(self, state):
        batch_size, u_w, h,w = state.shape
        U = state

        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        EU_x = convert_tensor((1 - U * self.axis_x.repeat(batch_size,u_w, 1, w))*U,device=self.device)
        EU_y = convert_tensor((1 - U * self.axis_y.repeat(batch_size,u_w, h, 1))*U,device=self.device)
        U_ = F.pad(U, pad=(1,1,1,1), mode='circular')
        Delta_u = F.conv2d(U_, self._laplacian)
        _eu_x = F.pad(EU_x, pad=(1,1,1,1), mode='circular')
        _eu_y = F.pad(EU_y, pad=(1,1,1,1), mode='circular')
        Div_eu_x = F.conv2d(_eu_x, self._grad_x)
        Div_eu_y = F.conv2d(_eu_y, self._grad_y)
        #Div_u = F.conv2d(U_, self._divergence)

        (D_1, D_2) = list(self.params.values())
        return D_1 * Delta_u + D_2 * (Div_eu_x+Div_eu_y)
    
    def _map_forward(self, state):
        batch_size, u_w, h,w = state.shape
        U = state

        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        EU_x = convert_tensor((1 - U * self.axis_x.repeat(batch_size,u_w,1,w))*U,device=self.device)
        EU_y = convert_tensor((1 - U * self.axis_y.repeat(batch_size,u_w, h, 1))*U,device=self.device)
        U_ = F.pad(U, pad=(1,1,1,1), mode='circular')
        Delta_u = F.conv2d(U_, self._laplacian)
        _eu_x = F.pad(EU_x, pad=(1,1,1,1), mode='circular')
        _eu_y = F.pad(EU_y, pad=(1,1,1,1), mode='circular')
        Div_eu_x = F.conv2d(_eu_x, self._grad_x)
        Div_eu_y = F.conv2d(_eu_y, self._grad_y)
        #Div_u = F.conv2d(U_, self._divergence)

        return {'delta_u':Delta_u, 'div_u':(Div_eu_x+Div_eu_y)}
    #    U = state

    #    U_ = F.pad(U, pad=(1,1,1,1), mode='circular')
    #    Delta_u = F.conv2d(U_, self._laplacian)
    #    Div_u = F.conv2d(U_, self._divergence)

    #    return {'delta_u':Delta_u,'div_u':(U * U * (1 + 1/4 * Div_u)-0.1*Div_u)}
        
    def get_value(self,x):
        batch_size, t_, u_w, h, w = x.shape
        x = x.contiguous()
        x = x.view(batch_size * t_, u_w, h, w)
        r = self._map_forward(x)
        x1 = r['delta_u']
        x2 = r['div_u']
        x1 = x1.view(batch_size, t_, u_w, h, w)
        x1 = x1.contiguous()
        x2 = x2.view(batch_size, t_, u_w, h, w)
        x2 = x2.contiguous()
        return {'delta_u':x1,'div_u':x2}

# nonparametric part
class NonParametricModel(nn.Module):
    def __init__(self, dx,state_dim=2, hidden=None,act=None,device="cpu",tx=32):
        super().__init__()
        if hidden is None:
            hidden = [16,64,64,16,1]
        if act is None:
            act = 'ReLU'
        self.layer_num = len(hidden)
        self.net1 = nn.Sequential()
        self.net1.add_module(str(0) + "linear" + 'net1', nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net1.add_module(str(i) + "Act" + 'net1', nn.ReLU())
            if act == 'Tanh':
                self.net1.add_module(str(i) + "Act" + 'net1', nn.Tanh())
            if act == 'Sine':
                self.net1.add_module(str(i) + "Act" + 'net1', siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net1.add_module(str(i+1) + "linear"+ 'net1',
                                nn.Linear(hidden[i], hidden[i+1],bias=True))
        self.net2 = nn.Sequential()
        self.net2.add_module(str(0) + "linear" + 'net2', nn.Linear(state_dim, hidden[0],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net2.add_module(str(i) + "Act" + 'net2', nn.ReLU())
            if act == 'Tanh':
                self.net2.add_module(str(i) + "Act" + 'net2', nn.Tanh())
            if act == 'Sine':
                self.net2.add_module(str(i) + "Act" + 'net2', siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net2.add_module(str(i+1) + "linear"+ 'net2',
                                nn.Linear(hidden[i], hidden[i+1],bias=True))
        self._initial_param()
        self.state_dim = state_dim
        self.device=device
        self._tx = tx
        
        # self.state_dim = state_dim
        # self.hidden = hidden
        # self.device=device
        # self.net1 = nn.Sequential(
        #     nn.Linear(self.state_dim,self.hidden[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[0],self.hidden[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[1],self.hidden[2]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[2],self.hidden[3]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[3],1)
        # )
        # self.net2 = nn.Sequential(
        #     nn.Linear(self.state_dim,self.hidden[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[0],self.hidden[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[1],self.hidden[2]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[2],self.hidden[3]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden[3],1)
        # )
        self._dx = dx
        self._grad_x = nn.Parameter(torch.tensor(
            [
                [ 0,  -1/2,  0],
                [ 0, 0,  0],
                [ 0,  1/2,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx), requires_grad=False)

        self._grad_y= nn.Parameter(torch.tensor(
            [
                [ 0,  0,  0],
                [ -1/2, 0,  1/2],
                [ 0,  0,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx), requires_grad=False)
    
        # axis_1dx = nn.Parameter(torch.tensor(range(64)).float()/32-63/64,requires_grad=False)
        # axis_1dy = nn.Parameter(torch.tensor(range(64)).float()/32-63/64,requires_grad=False)
        # self.axis_x = axis_1dx.view(1, 1, 64, 1)
        # self.axis_y = axis_1dy.view(1, 1, 1, 64)
        axis_1dx = nn.Parameter((2-torch.abs(torch.tensor(range(self._tx)).float()/self._tx*2-(self._tx-1)/self._tx))/2,requires_grad=False)
        axis_1dy = nn.Parameter((2-torch.abs(torch.tensor(range(self._tx)).float()/self._tx*2-(self._tx-1)/self._tx))/2,requires_grad=False)
        self.axis_x = convert_tensor(axis_1dx.view(1, 1, self._tx, 1),device=self.device)
        self.axis_y = convert_tensor(axis_1dy.view(1, 1, 1, self._tx),device=self.device)
        
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
        
    def forward(self,x):
        dim_x = len(x.shape)
        # for solution u
        if dim_x == 4:
            s = self._forward_4(x)
        elif dim_x == 5:
            s = self._forward_5(x)
        return s

    def _forward_5(self, x):
        batch_size, t_, u_w, h, w = x.shape
        x = x.permute(0,1,3,4,2).contiguous()
        x = x.view(batch_size*t_*h*w,u_w)
        #x_ = F.pad(x.view(batch_size*t_,u_w,h,w), pad=(1,1,1,1), mode='circular')
        #Div_x = F.conv2d(x_, self._divergence).view(batch_size, t_,u_w,h,w)
        y = convert_tensor(torch.cat([self.axis_x.repeat(batch_size,t_,1,1,w),self.axis_y.repeat(batch_size,t_,1,h,1)],dim=1),device=self.device)
        y = y.permute(0,1,3,4,2).contiguous()
        y = y.view(batch_size * t_ * h * w, self.state_dim)
        z1 = convert_tensor(self.net1(y)*x,device=self.device)
        z1 = z1.view(batch_size*t_, h, w, 1).permute(0,3,1,2).contiguous()
        z2 = convert_tensor(self.net2(y)*x,device=self.device)
        z2 = z2.view(batch_size*t_, h, w,1).permute(0,3,1,2).contiguous()
        result = self._reshape(z1, z2)
        result = result.view(batch_size, t_, u_w, h, w)

        return result

    def _forward_4(self, x):
        batch_size, u_w, h, w = x.shape
        #x_ = F.pad(x, pad=(1,1,1,1), mode='circular')
        #Div_x = F.conv2d(x_, self._divergence)
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(batch_size*h*w,u_w)
        y = convert_tensor(torch.cat([self.axis_x.repeat(batch_size,1,1,w),self.axis_y.repeat(batch_size,1,h,1)],dim=1),device=self.device)
        y = y.permute(0,2,3,1).contiguous()
        y = y.view(batch_size * h * w, self.state_dim)
        z1 = convert_tensor(self.net1(y)*x,device=self.device)
        z1 = z1.view(batch_size, h, w, 1).permute(0,3,1,2).contiguous()
        z2 = convert_tensor(self.net2(y)*x,device=self.device)
        z2 = z2.view(batch_size, h, w, 1).permute(0,3,1,2).contiguous()
        return self._reshape(z1, z2)

    # TODO Rename this here and in `forward`
    def _reshape(self, z1, z2):
        z1 = F.pad(z1, pad=(1,1,1,1), mode='circular')
        z2 = F.pad(z2, pad=(1,1,1,1), mode='circular')
        return F.conv2d(z1,self._grad_x) + F.conv2d(z2,self._grad_y)

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
            return self.para_model(state) + 0.1 * self.nonpara_model(state)

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
        return working_solution.permute(1,0,2,3,4).contiguous()

    def _L2_norm(self,x):
        return torch.linalg.vector_norm(x)#,dim=(0,1,3,4))

    def _L2_inner_product(self,x,y):
        return abs(torch.sum(x*y))#,dim=(0,1,3,4))

    def _penalty(self,para_x,nonpara_x):
        u_para_x1 = para_x['delta_u']
        u_para_x2 = para_x['div_u']
        u_nonpara_x = nonpara_x
        pen_u1 = self._L2_inner_product(u_para_x1,u_nonpara_x)/(self._L2_norm(u_para_x1)+1e-5)/(self._L2_norm(u_nonpara_x)+ 1e-5)
        pen_u2 = self._L2_inner_product(u_para_x2,u_nonpara_x)/(self._L2_norm(u_para_x2)+1e-5)/(self._L2_norm(u_nonpara_x)+ 1e-5)
        return 1/(1-pen_u1) + 1/(1-pen_u2) - 2

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