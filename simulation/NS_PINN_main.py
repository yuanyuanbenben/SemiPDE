'''Train equations using PINN.'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ignite.utils import convert_tensor
import GPUtil
from collections import OrderedDict

import argparse
import numpy as np
import copy

import PINN_model
import NS_dataset

# from utils import progress_bar, TemporaryGrad
parser = argparse.ArgumentParser(description='PyTorch PINN Training')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
parser.add_argument('--sigma','-sigma','-s',default = 0.1, type = float, help='variance of noise')
# parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--train_seed','-seed',default = 0, type = int, help='train seed')
parser.add_argument('--sample','-sample','-n',default = 64, type = int, help='sample size')
parser.add_argument('--lamda','-lamda','-l',default = 1.0, type = float, help='tuning parameter')
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# parameters setting

current_seed=args.train_seed
print('current seed %d'%(current_seed))
# setting parameters
D_1 = [6e-3,7e-3,8e-3]
D_2 = [3e-3,4e-3,5e-3]
func_f = None
func_g = None
bc = "periodic"
batch_size = [1,1]
train_curve_num= args.sample
test_curve_num= round(train_curve_num/4)
utest_curve_num= round(train_curve_num/4)    
sample_size=25
T_range=[0,2.5]
delta_t=2e-2
range_grid=[[-1,1],[-1,1],[-1,1]]
period=[True,True,True]
# num_grid=[16,16,16]
num_grid=[8,8,8]
seed_train=current_seed
seed_test=current_seed + 100
seed_utest=current_seed + 200
initial=[0,1]
noise_sigma=args.sigma
dim=3

# net
state_dim = [3,6]
hidden=[16,16]
lambda_0 = 0.1#5*64*25
tau_1 = 1e-3
tau_2 = 0.01
niter = 5
nupdate = 500
nepoch = 100
nlog= 100
dx = (range_grid[0][1]-range_grid[0][0])/num_grid[0]
path_train = "./data/ns2_train_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
path_test = "./data/ns2_test_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,seed_train)
path_utest = "./data/ns2_utest_%d_%d_%d_%d"%(train_curve_num,test_curve_num,
                                                                sample_size,seed_train)

path_checkpoint = "./checkpoint/ckpt_ns2_pinn_%d_%d_%d_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,seed_train)
if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    
# device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device='cpu'

# generate data
print('==> Preparing data..')
train,test,utest = NS_dataset.create_data(path_train=path_train,path_test=path_test,path_utest=path_utest,
        D_1=D_1,D_2=D_2,func_f=func_f,func_g = func_g,bc=bc,
        batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
        sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
        period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,seed_utest=seed_utest,
        initial=initial,noise_sigma=noise_sigma,dim=dim,rand=False)
print('==> Data already !')

print('==> Building model...')
state_dim_eq = 4
hidden_s = [32,32,3]
solution_net = dict()
for i in range(64):
    solution_net[str(i)] = PINN_model.PINN_Net(state_dim=state_dim_eq, hidden=hidden_s,act = 'Sine').to(device)
PDE_model = PINN_model.NSEquation(state_dim=state_dim, hidden=hidden).to(device)
print('==> Model download complete!')


# if device == 'cuda:0':
#     for i in range(64):
#         solution_net[str(i)].net = torch.nn.DataParallel(solution_net[str(i)].net)
#     cudnn.benchmark = True
    
# setting fixed seed
# torch.manual_seed(20230530)

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     if args.model == 'gfn':
#         checkpoint = torch.load('./checkpoint/ckpt_PINN_gfn.pth')
#         PDE_model.net.load_state_dict(checkpoint['PDE_model_net'])
#         PDE_model._params['D_1'] = checkpoint['D_1']
#         PDE_model._params['D_2'] = checkpoint['D_2']
#         solution_net.load_state_dict(checkpoint['solution_net'])
#         start_epoch = checkpoint['epoch']

criterion = nn.MSELoss()
solution_optimizer = dict()
for i in range(64):
    solution_optimizer[str(i)] = optim.Adam(solution_net[str(i)].parameters(), lr=2e-2, betas=(0.9, 0.999))
# solution_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(solution_optimizer, T_max=200)
PDE_model_optimizer = optim.Adam(PDE_model.parameters(), lr=args.lr, betas=(0.9,0.999))
# PDE_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PDE_model_optimizer, T_max=200)
t_range = np.linspace(T_range[0],T_range[1],sample_size + 1,endpoint=True)
x_range = np.linspace(-1+1/num_grid[0],1+1/num_grid[0],num_grid[0],endpoint=False) 
y_range = np.linspace(-1+1/num_grid[1],1+1/num_grid[1],num_grid[1],endpoint=False)
z_range = np.linspace(-1+1/num_grid[2],1+1/num_grid[2],num_grid[2],endpoint=False)
position = np.array([[t,x,y,z] for t in t_range for x in x_range for y in y_range for z in z_range])
position = torch.tensor(position,dtype=torch.float32).to(device)
position.requires_grad = True
# position_optimizer = optim.Adam(position_.parameters(), lr=0, betas=(0.9,0.999))
# position_x = np.array([[x,y,z] for t in t_range for x in x_range for y in y_range for z in z_range])
# position_x = torch.tensor(position_x,dtype=torch.float32).to(device)

class Optimize(object):
    def __init__(self,train,nepoch,solution_net,PDE_model,solution_optimizer,PDE_model_optimizer,lamda,position,device):
        self.train = train
        self.nepoch = nepoch
        self.solution_net = solution_net
        self.PDE_model = PDE_model
        self.solution_optimizer = solution_optimizer
        self.PDE_model_optimizer = PDE_model_optimizer
        self.lamda = lamda
        self.myloss = nn.MSELoss()
        self.position = position
        self.device = device

        
        
    def log(self,epoch, iteration, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(self.train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=self.nepoch,
            i=iteration+1,
            max_i=len(self.train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=value)
            
        print(message)
                
    # Training
    def train_func(self,epoch):
        print('\nEpoch: %d' % epoch)
        # for i in range(64):
        #     solution_net[str(i)].train()
        #     PDE_model.train()
        loss_test = 0
        loss_u = 0
        # print(1)
        # GPUtil.showUtilization(all=True,useOldCode=True)
        for iteration, states in enumerate(self.train):
            state = states['states']
            state = state.to(self.device)
            batch_, t_, u_w, h, w, l = state.shape
            state = state.permute(0,1,3,4,5,2).contiguous()
            state = state.view(batch_ * t_ * h * w * l, u_w)
            self.PDE_model(position,self.solution_net[str(iteration)])
            loss_eq = self.PDE_model.loss_eq 
            loss_data = self.myloss(self.solution_net[str(iteration)](position),state)
            self.loss = loss_eq * self.lamda + loss_data
            self.loss.backward()
            # print(2)
            # GPUtil.showUtilization(all=True,useOldCode=True)
            self.solution_optimizer[str(iteration)].step()
            self.PDE_model_optimizer.step()
            self.solution_optimizer[str(iteration)].zero_grad()
            self.PDE_model_optimizer.zero_grad()
            self.PDE_model.cache_out()
            loss_test += self.loss.item()
            loss_u += loss_data.item()
        # print(3)
        # GPUtil.showUtilization(all=True,useOldCode=True)
        loss_test = loss_test/(iteration + 1)
        loss_test = {
            'loss_test': loss_test,
        }
        loss_u = loss_u/(iteration + 1)
        loss_u = {
            'loss_u': loss_u - noise_sigma**2,
        }
        self.log(epoch, iteration, loss_test)
        self.log(epoch, iteration, loss_u)
        print(f'D_1:{self.PDE_model.params["D_1"]*1e3}e-3')
        print(f'D_2:{self.PDE_model.params["D_2"]*1e3}e-3')
        return loss_test['loss_test'],loss_u['loss_u']
    
    # # test
    # def val_func(self,epoch):
    #     # for i in range(64):
    #     #     solution_net[str(i)].train()
    #     #     PDE_model.train()
    #     loss_u = 0
    #     # print(1)
    #     # GPUtil.showUtilization(all=True,useOldCode=True)
    #     with torch.no_grad():
    #         for iteration, states in enumerate(self.utest):
    #             state = states['states']
    #             state = state.to(self.device)
    #             batch_, t_, u_w, h, w = state.shape
    #             state = state.permute(0,1,3,4,2).contiguous()
    #             state = state.view(batch_ * t_ * h * w, u_w)
    #             loss_data = self.myloss(self.solution_net[str(iteration)](self.position),state)
    #             loss_u += loss_data.item()
    #     # print(3)
    #     # GPUtil.showUtilization(all=True,useOldCode=True)
    #     loss_u = loss_u/(iteration + 1)
    #     loss_u = {
    #         'loss_u': loss_u,
    #     }
    #     print('#' * 80)
    #     self.log(epoch, iteration, loss_u)
    #     print('#' * 80)
        
    #     return loss_u['loss_u']
        
                

# def test(epoch):
#     print('Test\nEpoch: %d' % epoch)
#     if args.model == 'gfn':
#         global parameter_mat, parameter_mat2
#     solution_net.eval()
#     PDE_model.eval()
#     test_loss = 0
#     myloss = nn.MSELoss()
#     if args.model == 'gfn':
#         t_range = np.linspace(T_range[0],T_range[1],sample_size + 1,endpoint=True)
#         x_range = np.linspace(-1+0.125/2,1+0.125/2,16,endpoint=False) 
#         y_range = np.linspace(-1+0.125/2,1+0.125/2,16,endpoint=False)
#         position = np.array([[t,x,y] for idx in range(batch_size[1]) for t in t_range for x in x_range for y in y_range])
#     position = torch.tensor(position,dtype=torch.float32).to(device)
#     with torch.no_grad():
#         for batch_idx, states in enumerate(testset):
#             state = states['states']
#             state = convert_tensor(state, device)
#             if args.model == 'gfn':
#                 batch_, t_, u_w, h, w = state.shape
#                 state = state.permute(0,1,3,4,2).contiguous()
#                 state = state.view(batch_ * t_ * h * w, u_w)
#             loss = myloss(solution_net(position),state)
#             test_loss += loss.item()

#             progress_bar(batch_idx, len(testset), 'Average Loss: %.3f | Current Loss: %.3f'
#                         % (test_loss/(batch_idx+1), loss.item()))
#     print(f'D_1:{PDE_model.params["D_1"]*1e3}e-3')
#     print(f'D_2:{PDE_model.params["D_2"]*1e3}e-3')

    # # Save checkpoint.
    # loss = test_loss/(batch_idx+1)
    # parameter_mat.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_1"])).numpy().copy())
    # parameter_mat2.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_2"])).numpy().copy())

# if __name__ == '__main__':
tor = 5e-2
best_loss = torch.inf
# if args.model == 'gfn':
#     parameter_mat = []
#     parameter_mat2 = []
train_process = Optimize(train,nepoch,solution_net,PDE_model,solution_optimizer,PDE_model_optimizer,args.lamda,position,device)
for epoch in range(nepoch):
    loss, loss_u = train_process.train_func(epoch)
    if loss < best_loss:
        print('Saving..')
        state = {
                'solution_net': train_process.solution_net[str(0)].state_dict(),
                'PDE_model_net1': train_process.PDE_model.net1.state_dict(),
                'PDE_model_net2': train_process.PDE_model.net2.state_dict(),
                'D_1': train_process.PDE_model.params['D_1'],
                'D_2': train_process.PDE_model.params['D_2'],
                'u_loss': loss_u,
                'loss':loss,
                'epoch': epoch,
                'seed':torch.initial_seed()
        }
        torch.save(state, path_checkpoint)
        best_loss = loss
    # if (epoch % 5 == 4):
    #     test(epoch)
    # solution_scheduler.step()
    # PDE_model_scheduler.step()
# save_mat = np.array([parameter_mat,parameter_mat2])
# np.save(path_save,save_mat)
