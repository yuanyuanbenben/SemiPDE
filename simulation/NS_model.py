# neural network approximate for f(u,w) and g(u,w)
import torch
import torch.nn as nn
import siren_pytorch as siren
from collections import OrderedDict
import torch.nn.functional as F
from torchdiffeq import odeint
from ignite.utils import convert_tensor
import pde

# parametric part
class ParametricPart(nn.Module):
    def __init__(self, dx=8, device='cpu'):
        super().__init__()
        self._dx = dx
        self.device = device
        self._laplacian = nn.Parameter(torch.tensor(
            [
            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]],

            [[0, 1, 0],
            [1, -6, 1],
            [0, 1, 0]],

            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]
            ],
        ).float().view(1, 1, 3, 3, 3) / (self._dx * self._dx), requires_grad=False)

        self._grad = nn.Parameter(torch.tensor(
            [-1/2,0,1/2],
        ).float().view(1, 1, 3) / (self._dx), requires_grad=False)

        self._params = nn.ParameterDict({
            'D_1': nn.Parameter(torch.tensor([0.,0.,0.])), 
            'D_2': nn.Parameter(torch.tensor([0.,0.,0.])),
        })
        # self.grid = pde.CartesianGrid([[-1,1],[-1,1],[-1,1]], [16,16,16], [True,True,True])
        self.grid = pde.CartesianGrid([[-1,1],[-1,1],[-1,1]], [8,8,8], [True,True,True])
        # self._p = nn.Parameter(torch.tensor(
        #     pde.VectorField.from_expression(self.grid,["1-abs(x)","1-abs(y)","1-abs(z)"]).data
        # ).float().view(1,3,16,16,16),requires_grad=False)
        self._p = nn.Parameter(torch.tensor(
            pde.VectorField.from_expression(self.grid,["1-abs(x)","1-abs(y)","1-abs(z)"]).data
        ).float().view(1,3,8,8,8),requires_grad=False)
        self.params = OrderedDict()
    
    def _gradients(self, u):
        batch_size, u_w, h, w, l = u.shape    
        if u_w == 1:
            # return the gradients
            # batch_size, u_w, h, w, l
            return convert_tensor(
                torch.cat((
                    F.conv1d(F.pad(u.permute(0,3,4,1,2).contiguous().view(batch_size*w*l*u_w,1,h),pad=(1,1),mode='circular'
                    ),self._grad).view(batch_size,w,l,u_w,h).contiguous().permute(0,3,4,1,2),
                    F.conv1d(F.pad(u.permute(0,2,4,1,3).contiguous().view(batch_size*h*l*u_w,1,w),pad=(1,1),mode='circular'
                    ),self._grad).view(batch_size,h,l,u_w,w).contiguous().permute(0,3,1,4,2),
                    F.conv1d(F.pad(u.permute(0,2,3,1,4).contiguous().view(batch_size*h*w*u_w,1,l),pad=(1,1),mode='circular'
                    ),self._grad).view(batch_size,h,w,u_w,h).contiguous().permute(0,3,1,2,4)
                ),
                dim=1),
            device = self.device
            )
        elif u_w == 3:
            # return the gradients
            # batch_size u_w u_w h w l
            return convert_tensor(
                torch.cat((
                    F.conv1d(F.pad(u.permute(0,3,4,1,2).contiguous().view(batch_size*w*l*u_w,1,h),pad=(1,1),mode='circular'
                    ),self._grad).view(batch_size,w,l,u_w,1,h).contiguous().permute(0,3,4,5,1,2),
                    F.conv1d(F.pad(u.permute(0,2,4,1,3).contiguous().view(batch_size*h*l*u_w,1,w),pad=(1,1),mode='circular'
                    ),self._grad).view(batch_size,h,l,u_w,1,w).contiguous().permute(0,3,4,1,5,2),
                    F.conv1d(F.pad(u.permute(0,2,3,1,4).contiguous().view(batch_size*h*w*u_w,1,l),pad=(1,1),mode='circular'
                    ),self._grad).view(batch_size,h,w,u_w,1,h).contiguous().permute(0,3,4,1,2,5)
                ),
                dim=2),
            device = self.device
            )


    def forward(self, state):
        batch_size, u_w, h,w,l = state.shape
        U = state

        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        grad_u  = self._gradients(U).permute(0,3,4,5,1,2).contiguous().view(batch_size*h*w*l,u_w,u_w)
        grad_p = self._gradients(torch.sum(U*self._p,dim=1,keepdim=True))
        u_grad_u = convert_tensor(torch.matmul(U.permute(0,2,3,4,1).contiguous().view(batch_size*h*w*l,1,u_w), grad_u
        ).view(batch_size,h,w,l,u_w).contiguous().permute(0,4,1,2,3),device=self.device)
        Delta_u = F.conv3d(F.pad(U.contiguous().view(batch_size*u_w,1,h,w,l), pad=(1,1,1,1,1,1), mode='circular'), self._laplacian).view(batch_size,u_w,h,w,l,)
        grad_div_u = self._gradients(torch.sum(torch.diagonal(grad_u,dim1=1,dim2=2),dim=1).view(batch_size,1,h,w,l))
        (D_1, D_2) = list(self.params.values())
        return - 1e-3*u_grad_u + D_1.view(1,3,1,1,1) * (Delta_u - 1/3 * grad_div_u) - D_2.view(1,3,1,1,1) * grad_p
    
    def _map_forward(self, state):
        batch_size, u_w, h,w,l = state.shape
        U = state

        self.params['D_1'] = self._params['D_1']
        self.params['D_2'] = self._params['D_2']
        grad_u  = self._gradients(U).permute(0,3,4,5,1,2).contiguous().view(batch_size*h*w*l,u_w,u_w)
        grad_p = self._gradients(torch.sum(U*self._p,dim=1,keepdim=True))
        u_grad_u = convert_tensor(torch.matmul(U.permute(0,2,3,4,1).contiguous().view(batch_size*h*w*l,1,u_w), grad_u
        ).view(batch_size,h,w,l,u_w).contiguous().permute(0,4,1,2,3),device=self.device)
        Delta_u = F.conv3d(F.pad(U.contiguous().view(batch_size*u_w,1,h,w,l), pad=(1,1,1,1,1,1), mode='circular'), self._laplacian).view(batch_size,u_w,h,w,l)
        grad_div_u = self._gradients(torch.sum(torch.diagonal(grad_u,dim1=1,dim2=2),dim=1).view(batch_size,1,h,w,l))

        return {'u_grad_u':u_grad_u, 'delta_u':(Delta_u - 1/3 * grad_div_u), 'grad_p':grad_p}

    def get_value(self,x):
        batch_size, t_, u_w, h, w, l = x.shape
        x = x.contiguous()
        x = x.view(batch_size * t_, u_w, h, w,l)
        r = self._map_forward(x)
        x1 = r['u_grad_u']
        x2 = r['delta_u']
        x3 = r['grad_p']
        x1 = x1.view(batch_size, t_, u_w, h, w,l)
        x1 = x1.contiguous()
        x2 = x2.view(batch_size, t_, u_w, h, w,l)
        x2 = x2.contiguous()
        x3 = x3.view(batch_size, t_, u_w, h, w,l)
        x3 = x3.contiguous()
        return {'u_grad_u':x1, 'delta_u':x2, 'grad_p':x3}

# nonparametric part
class NonParametricModel(ParametricPart):
    def __init__(self, dx, state_dim=None, hidden=None, device="cpu"):
        super(NonParametricModel, self).__init__()
        if state_dim is None:
            state_dim = [3,6]
        if hidden is None:
            hidden = [16,64,64,16]
        self.state_dim = state_dim
        self.hidden = hidden
        self.device=device
        # net for f_1(x)
        self.net1 = nn.Sequential(
            nn.Linear(self.state_dim[0],self.hidden[0]),
            nn.ReLU(),
            nn.Linear(self.hidden[0],self.hidden[1]),
            nn.ReLU(),
            nn.Linear(self.hidden[1],self.hidden[2]),
            nn.ReLU(),
            nn.Linear(self.hidden[2],self.hidden[3]),
            nn.ReLU(),
            nn.Linear(self.hidden[3],3)
        )
        # net for f_2(x,u)
        self.net2 = nn.Sequential(
            nn.Linear(self.state_dim[1],self.hidden[0]),
            nn.ReLU(),
            nn.Linear(self.hidden[0],self.hidden[1]),
            nn.ReLU(),
            nn.Linear(self.hidden[1],self.hidden[2]),
            nn.ReLU(),
            nn.Linear(self.hidden[2],self.hidden[3]),
            nn.ReLU(),
            nn.Linear(self.hidden[3],1)
        )
        self._dx = dx
        # self._x = nn.Parameter(torch.tensor(
        #     pde.VectorField.from_expression(self.grid,["x","y","z"]).data
        # ).float().view(1,3,16,16,16),requires_grad=False)
        self._x = nn.Parameter(torch.tensor(
            pde.VectorField.from_expression(self.grid,["x","y","z"]).data
        ).float().view(1,3,8,8,8),requires_grad=False)



    def forward(self,x):
        dim_x = len(x.shape)

        # for solution u
        if dim_x == 5:
            batch_size, u_w, h, w,l = x.shape
            grad_u  = self._gradients(x.view(batch_size,u_w,h,w,l)).permute(0,3,4,5,1,2).contiguous().view(batch_size*h*w*l,u_w,u_w)
            x = x.permute(0,2,3,4,1).contiguous()
            x = x.view(batch_size*h*w*l,u_w)
            axis_x = convert_tensor(self._x.repeat(batch_size,1,1,1,1
                ).permute(0,2,3,4,1).contiguous().view(batch_size*h*w*l,3),device=self.device)
            y = torch.cat((axis_x,x),dim=1)
            z1 = convert_tensor(self.net1(axis_x),device=self.device)
            z1 = z1.view(batch_size, h, w,l, 3).contiguous().permute(0,4,1,2,3)
            div_u = convert_tensor(torch.sum(torch.diagonal(grad_u,dim1=1,dim2=2),dim=1,keepdim=True),device=self.device)
            z2 = self._gradients((self.net2(y)*div_u).contiguous().view(batch_size,1,h,w,l))

        elif dim_x == 6:
            batch_size, t_, u_w, h, w,l= x.shape
            grad_u  = self._gradients(x.view(batch_size*t_,u_w,h,w,l)).permute(0,3,4,5,1,2).contiguous().view(batch_size*t_*h*w*l,u_w,u_w)
            x = x.permute(0,1,3,4,5,2).contiguous()
            x = x.view(batch_size*t_*h*w*l,u_w)
            axis_x = convert_tensor(self._x.repeat(batch_size*t_,1,1,1,1
                ).permute(0,2,3,4,1).contiguous().view(batch_size*t_*h*w*l,3),device=self.device)
            y = torch.cat((axis_x,x),dim=1)
            z1 = convert_tensor(self.net1(axis_x),device=self.device)
            z1 = z1.view(batch_size,t_, h, w,l, 3).contiguous().permute(0,1,5,2,3,4)
            div_u = convert_tensor(torch.sum(torch.diagonal(grad_u,dim1=1,dim2=2),dim=1,keepdim=True),device=self.device)
            z2 = self._gradients((self.net2(y)*div_u).contiguous().view(batch_size*t_,1,h,w,l)).contiguous().view(batch_size,t_,u_w,h,w,l)

        return z1 + z2

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
        return working_solution.permute(1,0,2,3,4,5).contiguous()

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
