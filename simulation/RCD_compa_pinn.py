'''Train equations using PINN.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ignite.utils import convert_tensor

import os
import argparse
import numpy as np
import copy

import PINN_model
import PINN_dataset

from utils import progress_bar, TemporaryGrad
parser = argparse.ArgumentParser(description='PyTorch PINN Training')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = torch.device("cuda:0")
# device = 'cuda'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# parameters setting
D_1 = 1
D_2 = 0.1
func_f = None
bc = "periodic"
batch_size = [84,84]
train_curve_num= 1
test_curve_num= 1
sample_size=20
T_range=[0,20]
delta_t=5e-4
range_grid=[[0,40]]
period=[True]
num_grid=[40]
seed_train=None
seed_test=20221106
noise_sigma=0.05 # 0.02/0.05
dim=1
# net
state_dim = 1
hidden=[16,64,16,dim]
lambda_0 = 1#5*64*25
tau_1 = 5e-3
tau_2 = 1
niter = 5
nupdate = 50
nepoch = 200
nlog= 10
dx = (range_grid[0][1]-range_grid[0][0])/num_grid[0]
path_train = "./data/rcd_train_PINN_%d_%d_%d_%.3f_%.3f_%.3f"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2)
path_test = "./data/rcd_test_PINN_%d_%d_%d_%.3f_%.3f_%.3f"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2)

mode = 'nonpara'

if mode == 'para':
    path_save = "./outputs/rcd_PINN_para_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2)

    path_checkpoint = "./checkpoint/ckpt_rcd_PINN_para_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2)
else:
    path_save = "./outputs/rcd_PINN_nonpara_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2)

    path_checkpoint = "./checkpoint/ckpt_rcd_PINN_nonpara_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2)

if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    
# generate data
print('==> Preparing data..')
trainset,testset = PINN_dataset.create_pinn_data(
    D_1=D_1,D_2=D_2,func_f=func_f,bc=bc,
    batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
    sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
    period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,noise_sigma=noise_sigma,dim=dim)
print('==> Data already !')


    
print('==> Building model...')
state_dim = 2
hidden_s = [32,128,512,512,128,32,1]
# lamda1 = 0.001 for sigma=0.05 and 0.001 for sigma=0.02
lamda1 = 0.01
solution_net = PINN_model.PINN_Net(state_dim=state_dim, hidden=hidden_s,device=device)
# solution_net._initial_param()
if mode == 'para':
    PDE_model = PINN_model.ParaGCDEquation()
else:
    PDE_model = PINN_model.NonParaGCDEquation(state_dim=dim, hidden=hidden,lamda = lamda1, device=device)
print('==> Model downlod complete!')

# setting fixed seed
torch.manual_seed(20230530)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if mode == 'para':
        checkpoint = torch.load("./checkpoint/ckpt_rcd_PINN_para_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2))
    else:
        checkpoint = torch.load("./outputs/rcd_PINN_nonpara_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                sample_size,noise_sigma,D_1,D_2))
        PDE_model.net.load_state_dict(checkpoint['PDE_model_net'])
        
    PDE_model._params['D_1'] = checkpoint['D_1']
    PDE_model._params['D_2'] = checkpoint['D_2']
    PDE_model._params['D_3'] = checkpoint['D_3']
    solution_net.load_state_dict(checkpoint['solution_net'])
    start_epoch = checkpoint['epoch']

criterion = nn.MSELoss()

if mode == 'para':
    solution_optimizer = optim.Adam(solution_net.parameters(), lr=args.lr, betas=(0.9,0.999))
    solution_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(solution_optimizer, T_max=5000)
    PDE_model_optimizer = optim.Adam(PDE_model.parameters(), lr=args.lr, betas=(0.9,0.999))
    PDE_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PDE_model_optimizer, T_max=5000)
else:
    solution_optimizer = optim.Adam(solution_net.parameters(), lr=args.lr, betas=(0.9,0.999))
    solution_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(solution_optimizer, T_max=20000)
    PDE_model_optimizer = optim.Adam(PDE_model.parameters(), lr=args.lr, betas=(0.9,0.999))
    PDE_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PDE_model_optimizer, T_max=20000)

# Training
def train(epoch):
    if epoch > 500:
        lamda2 = 10
    else:
        lamda2 = 0
    global batch_size
    global device
    print('\nEpoch: %d' % epoch)
    solution_net.train()
    PDE_model.train()
    train_loss = 0
    myloss = nn.MSELoss()
    # t_range = np.linspace(T_range[0],T_range[1],sample_size + 1,endpoint=True)
    # x_range = np.linspace(0.05,40.05,400,endpoint=False) 
    # position = np.ndarray((1,21,2,400))
    # init_position = [[0,x] for x in np.linspace(0.05,40.05,400,endpoint=False) ]
    # for i in range(21):
    #     position[0,i,0,:] = i
    #     position[0,i,1,:] = x_range
    # position = torch.tensor(position,dtype=torch.float32).to(device)
    # init_position = torch.tensor(init_position,dtype=torch.float32).to(device)
    for batch_idx, states in enumerate(trainset):
        state = states['states']
        # print(state)
        position = states['position']
        in_position = states['in_position']
        # print(position)
        boundary1 = states['boundary1']
        boundary2 = states['boundary2']
        state = convert_tensor(state, device)
        position = convert_tensor(position, device)
        in_position = convert_tensor(in_position, device)
        boundary1 = convert_tensor(boundary1,device)
        boundary2 = convert_tensor(boundary2, device)
        # batch_, t_, u_w, h = state.shape
        # state = state.permute(0,1,3,2).contiguous()
        # state = state.view(batch_ * t_ * h, u_w)
        # #print(state)
        # position = position.permute(0,1,3,2).contiguous()
        # position = position.view(batch_ * t_ * h, 2)
        #print(position)
        loss_eq = PDE_model(in_position,solution_net)['loss_eq']
        pred = solution_net(position)
        # print(pred)
        # print(pred.view( batch_, t_, h))
        loss_data = myloss(pred,state)
        loss = loss_data + loss_eq * lamda2 + myloss(solution_net(boundary1),solution_net(boundary2)) * 10
        loss.backward()
        solution_optimizer.step()
        PDE_model_optimizer.step()
        train_loss += loss.item()

        progress_bar(batch_idx, len(trainset), 'Average Loss: %.3f | Current Loss: %.3f'
                    % (train_loss/(batch_idx+1), loss.item()))
    
                

def test(epoch):
    global mode
    global batch_size
    global best_loss
    global device
    global path_checkpoint
    print('Test\nEpoch: %d' % epoch)
    global test_loss_mat, parameter_mat, parameter_mat2, parameter_mat3
    solution_net.eval()
    PDE_model.eval()
    test_loss = 0
    myloss = nn.MSELoss()
    # t_range = np.linspace(T_range[0],T_range[1],sample_size + 1,endpoint=True)
    # x_range = np.linspace(0.05,40.05,400,endpoint=False) 
    # position = np.array([[t,x] for idx in range(batch_size[0]) for t in t_range for x in x_range])
    # position = torch.tensor(position,dtype=torch.float32).to(device)
    with torch.no_grad():
        for batch_idx, states in enumerate(testset):
            state = states['states']
            position = states['position']
            state = convert_tensor(state, device)
            position = convert_tensor(position, device)
            # batch_, t_, u_w, h = state.shape
            # state = state.permute(0,1,3,2).contiguous()
            # state = state.view(batch_ * t_ * h, u_w)
            # position = position.permute(0,1,3,2).contiguous()
            # position = position.view(batch_ * t_ * h, 2)
            loss = myloss(solution_net(position),state) #+ PDE_model(position,solution_net)['loss_eq']
            test_loss += loss.item()

            progress_bar(batch_idx, len(testset), 'Average Loss: %.3f | Current Loss: %.3f'
                        % (test_loss/(batch_idx+1), loss.item()))
    print(f'D_1:{PDE_model.params["D_1"]}')
    print(f'D_2:{PDE_model.params["D_2"]}')
    print(f'D_3:{PDE_model.params["D_3"]}')

    # Save checkpoint.
    loss = test_loss/(batch_idx+1)
    test_loss_mat.append(loss)
    parameter_mat.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_1"])).numpy().copy())
    parameter_mat2.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_2"])).numpy().copy())
    parameter_mat3.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_3"])).numpy().copy())
    if loss < best_loss:
        print('Saving..')
        if mode == 'para':
            state = {
                    'solution_net': solution_net.state_dict(),
                    'D_1': PDE_model.params['D_1'],
                    'D_2': PDE_model.params['D_2'],
                    'D_3': PDE_model.params['D_3'],
                    'loss': loss,
                    'epoch': epoch,
                    'seed':torch.initial_seed()
            }
        else: 
            state = {
                    'solution_net': solution_net.state_dict(),
                    'PDE_model_net': PDE_model.net.state_dict(),
                    'D_1': PDE_model.params['D_1'],
                    'D_2': PDE_model.params['D_2'],
                    'D_3': PDE_model.params['D_3'],
                    'loss': loss,
                    'epoch': epoch,
                    'seed':torch.initial_seed()
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, path_checkpoint)
        best_loss = loss

# if __name__ == '__main__':
tor = 5e-2
best_loss = torch.inf
parameter_mat = []
parameter_mat2 = []
parameter_mat3 = []
test_loss_mat = []
if mode == 'para':
    epoch_num = 5000
else:
    epoch_num = 20000
    
for epoch in range(start_epoch, start_epoch+epoch_num):
    train(epoch)
    if (epoch % 100 == 99):
        test(epoch)
    solution_scheduler.step()
    PDE_model_scheduler.step()
save_mat = np.array([test_loss_mat,parameter_mat,parameter_mat2,parameter_mat3])
np.save(path_save,save_mat)
        
        