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

from utils import progress_bar, TemporaryGrad
parser = argparse.ArgumentParser(description='PyTorch PINN Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model','-m',default='gfn',type=str,help='model')
args = parser.parse_args()

device = torch.device("cuda:1")
# device = 'cuda'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# parameters setting

if args.model == 'gfn':
    import GFN_dataset
    D_1 = 1e-3
    D_2 = 5e-3                                                    
    func_f = lambda u,w: u-u**3-5e-3-w
    func_g = None
    #func_f = lambda u,w: 0
    #func_g = lambda u,w: 0
    a = 0.1
    epsilon = 1
    gamma = 1 
    bc = None
    batch_size = [16,4]
    train_curve_num= 64
    test_curve_num= 16
    sample_size = 50
    T_range=[0,2.5]
    delta_t=1e-3
    range_grid=[[-1,1],[-1,1]]
    period=[True,True]
    num_grid=[16,16]
    seed_train=20221106
    seed_test=20221106
    initial=[0,1]
    noise_sigma=0.1
    dim=2
    hidden=[16,64,64,16,dim]
    lambda_0 = 1.0
    tau_1 = 1e-3
    tau_2 = 1
    niter = 5
    nupdate = 20
    nepoch = 100
    nlog= 4
    path_train = "./data/gfn_train_PINN_%d_%d_%d_%.3f_%.3f_%.3f"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)
    path_test = "./data/gfn_test_PINN_%d_%d_%d_%.3f_%.3f_%.3f"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2)
    path_save = "./outputs/gfn_PINN_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2)

    path_checkpoint = "./checkpoint/ckpt_gfn_PINN_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2)
    tor_1 = 1e-4
    tor_2 = 1e-4
    iteration= 30
    pertu_loss= 1e-5
    pertu_grad= 1e-5
    stepsize= 1e-5
    epsilon= 1e-2
    do = True
    arti_score = True
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    trainset, testset = GFN_dataset.create_data(path_train=path_train,path_test=path_test,
        D_1=D_1,D_2=D_2,func_f=func_f,func_g=func_g,a=a,epsilon=epsilon,gamma=gamma,bc=bc,
        batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
        sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
        period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,
        initial=initial,noise_sigma=noise_sigma,dim=dim,device = device)
    print('==> Data already !')

    
    print('==> Building model...')
    state_dim = 3
    hidden_s = [32,128,128,32,2]
    lamda = 0.1
    solution_net = PINN_model.PINN_Net(state_dim=state_dim, hidden=hidden_s,act = 'Sine',device=device).to(device)
    PDE_model = PINN_model.GFNEquation(T_range=T_range, range_grid=range_grid, period=period, dt = 2.5/50,
                                       state_dim=dim, hidden=hidden,device=device).to(device)
    
    print('==> Model downlod complete!')


if device == 'cuda':
    solution_net = torch.nn.DataParallel(solution_net)
    PDE_model = torch.nn.DataParallel(PDE_model)
    cudnn.benchmark = True
    
# setting fixed seed
torch.manual_seed(20230530)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.model == 'gfn':
        checkpoint = torch.load('./checkpoint/ckpt_PINN_gfn.pth')
        PDE_model.net.load_state_dict(checkpoint['PDE_model_net'])
        PDE_model._params['D_1'] = checkpoint['D_1']
        PDE_model._params['D_2'] = checkpoint['D_2']
        solution_net.load_state_dict(checkpoint['solution_net'])
        start_epoch = checkpoint['epoch']

criterion = nn.MSELoss()
solution_optimizer = optim.Adam(solution_net.parameters(), lr=args.lr, betas=(0.9,0.999))
solution_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(solution_optimizer, T_max=200)
PDE_model_optimizer = optim.Adam(PDE_model.parameters(), lr=args.lr, betas=(0.9,0.999))
PDE_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PDE_model_optimizer, T_max=200)


# Training
def train(epoch):
    global batch_size
    global device
    print('\nEpoch: %d' % epoch)
    solution_net.train()
    PDE_model.train()
    train_loss = 0
    myloss = nn.MSELoss()
    if args.model == 'gfn':
        t_range = np.linspace(T_range[0],T_range[1],sample_size + 1,endpoint=True)
        x_range = np.linspace(-1+0.125/2,1+0.125/2,16,endpoint=False) 
        y_range = np.linspace(-1+0.125/2,1+0.125/2,16,endpoint=False)
        position = np.array([[t,x,y] for idx in range(batch_size[0]) for t in t_range for x in x_range for y in y_range])
    position = torch.tensor(position,dtype=torch.float32).to(device)
    for batch_idx, states in enumerate(trainset):
        state = states['states']
        state = convert_tensor(state, device)
        if args.model == 'gfn':
            batch_, t_, u_w, h, w = state.shape
            state = state.permute(0,1,3,4,2).contiguous()
            state = state.view(batch_ * t_ * h * w, u_w)
        loss_eq = PDE_model(position,solution_net)['loss_eq']
        loss_data = myloss(solution_net(position),state)
        loss = loss_data + loss_eq * lamda
        loss.backward()
        solution_optimizer.step()
        PDE_model_optimizer.step()
        train_loss += loss.item()

        progress_bar(batch_idx, len(trainset), 'Average Loss: %.3f | Current Loss: %.3f'
                    % (train_loss/(batch_idx+1), loss.item()))
    
                

def test(epoch):
    global batch_size
    global best_loss
    global device
    global path_checkpoint
    print('Test\nEpoch: %d' % epoch)
    if args.model == 'gfn':
        global parameter_mat, parameter_mat2
    solution_net.eval()
    PDE_model.eval()
    test_loss = 0
    myloss = nn.MSELoss()
    if args.model == 'gfn':
        t_range = np.linspace(T_range[0],T_range[1],sample_size + 1,endpoint=True)
        x_range = np.linspace(-1+0.125/2,1+0.125/2,16,endpoint=False) 
        y_range = np.linspace(-1+0.125/2,1+0.125/2,16,endpoint=False)
        position = np.array([[t,x,y] for idx in range(batch_size[1]) for t in t_range for x in x_range for y in y_range])
    position = torch.tensor(position,dtype=torch.float32).to(device)
    with torch.no_grad():
        for batch_idx, states in enumerate(testset):
            state = states['states']
            state = convert_tensor(state, device)
            if args.model == 'gfn':
                batch_, t_, u_w, h, w = state.shape
                state = state.permute(0,1,3,4,2).contiguous()
                state = state.view(batch_ * t_ * h * w, u_w)
            loss = myloss(solution_net(position),state)
            test_loss += loss.item()

            progress_bar(batch_idx, len(testset), 'Average Loss: %.3f | Current Loss: %.3f'
                        % (test_loss/(batch_idx+1), loss.item()))
    print(f'D_1:{PDE_model.params["D_1"]*1e3}e-3')
    print(f'D_2:{PDE_model.params["D_2"]*1e3}e-3')

    # Save checkpoint.
    loss = test_loss/(batch_idx+1)
    parameter_mat.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_1"])).numpy().copy())
    parameter_mat2.append(torch.Tensor.detach(torch.Tensor.cpu(PDE_model.params["D_2"])).numpy().copy())
    if loss < best_loss:
        print('Saving..')
        state = {
                'solution_net': solution_net.state_dict(),
                'PDE_model_net': PDE_model.net.state_dict(),
                'D_1': PDE_model.params['D_1'],
                'D_2': PDE_model.params['D_2'],
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
if args.model == 'gfn':
    parameter_mat = []
    parameter_mat2 = []
for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    if (epoch % 5 == 4):
        test(epoch)
    solution_scheduler.step()
    PDE_model_scheduler.step()
save_mat = np.array([parameter_mat,parameter_mat2])
np.save(path_save,save_mat)
        
        