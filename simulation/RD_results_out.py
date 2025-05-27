import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import RD_model


import argparse
parser = argparse.ArgumentParser(description='output')
parser.add_argument('--model_mode','-model','-mod',default = None, type = str, help='model type')
parser.add_argument('--sigma','-sigma','-s',default = 0.1, type = float, help='variance of noise')
parser.add_argument('--sample_size','-size','-ss',default = 800, type = int, help='sample size')
args = parser.parse_args()

dim=1
# hidden=[16,64,64,16,dim]
# test_x = np.linspace(0,1,1001,endpoint=True).reshape(1001,1)
# test_x = torch.tensor(test_x).float()
# true_f = lambda u: u*(1-u)
para_mat = np.ndarray((1,50))
# nonpara_mat = np.ndarray((1001,50))
u_mat = np.ndarray((1,50))
noise = args.sigma
sample_size = args.sample_size
model = args.model_mode
for i in range(50):
    if sample_size == 800:
        if model == 'semipde':
            checkpoints = torch.load('checkpoint/ckpt_rd2_64_16_25_%.3f_0.003_%d.pth'%(noise,i))
            # model_net = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
            # model_net.net.load_state_dict(checkpoints['net'])
            para_mat[:,i] = checkpoints['D'].to('cpu').detach().numpy()*1e-3
            # nonpara_mat[:,i:(i+1)] = (model_net.net(test_x) - true_f(test_x)).detach().to('cpu').numpy()
            u_mat[:,i] = checkpoints['u_loss']
        if model == 'baseline1':
            checkpoints = torch.load('checkpoint/ckpt_rd2_incom_64_16_25_%.3f_0.003_%d.pth'%(noise,i))
            # model_net = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
            # model_net.net.load_state_dict(checkpoints['net'])
            para_mat[:,i] = checkpoints['D'].to('cpu').detach().numpy()*1e-3
            # nonpara_mat[:,i:(i+1)] = (model_net.net(test_x) - true_f(test_x)).detach().to('cpu').numpy()
            u_mat[:,i] = checkpoints['u_loss']
        if model == 'baseline2':
            checkpoints = torch.load('checkpoint/ckpt_rd2_nonpara_64_16_25_%.3f_0.003_%d.pth'%(noise,i))
            u_mat[:,i] = checkpoints['u_loss']
    if sample_size == 1600:
        if model == 'semipde':
            checkpoints = torch.load('checkpoint/ckpt_rd_64_16_25_%.3f_0.003_%d.pth'%(noise,i))
            # model_net = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
            # model_net.net.load_state_dict(checkpoints['net'])
            para_mat[:,i] = checkpoints['D'].to('cpu').detach().numpy()*1e-3
            # nonpara_mat[:,i:(i+1)] = (model_net.net(test_x) - true_f(test_x)).detach().to('cpu').numpy()
            u_mat[:,i] = checkpoints['u_loss']
        if model == 'baseline1':
            checkpoints = torch.load('checkpoint/ckpt_rd_incom_64_16_25_%.3f_0.003_%d.pth'%(noise,i))
            # model_net = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
            # model_net.net.load_state_dict(checkpoints['net'])
            para_mat[:,i] = checkpoints['D'].to('cpu').detach().numpy()*1e-3
            # nonpara_mat[:,i:(i+1)] = (model_net.net(test_x) - true_f(test_x)).detach().to('cpu').numpy()
            u_mat[:,i] = checkpoints['u_loss']
        if model == 'baseline2':
            checkpoints = torch.load('checkpoint/ckpt_rd_nonpara_64_16_25_%.3f_0.003_%d.pth'%(noise,i))
            u_mat[:,i] = checkpoints['u_loss']
            
if model == 'semipde':
    bias = np.mean(para_mat,axis=1) - 3e-3
    std = np.sqrt(np.mean(np.square(para_mat - np.mean(para_mat,axis=1)),axis=1))
    rmse = np.sqrt(bias**2 + std**2)
    # bias_non = np.sqrt(np.mean(np.square(np.mean(nonpara_mat,axis=1))))
    # std_non = np.sqrt(np.mean(np.square(nonpara_mat -np.mean(nonpara_mat,axis=1,keepdims=True))))
    # rmse_non = np.sqrt(bias_non**2 + std_non**2)
    rmse_u = np.sqrt(np.mean(u_mat,axis=1))
    print('For semipde:')
    print('bias for para:%.3fe-2'%(bias*1e2))
    print('std for para:%.3fe-2'%(std*1e2))
    print('rmse for para:%.3fe-2'%(rmse*1e2))
    # print('bias for nonpara func:%.6f'%(bias_non))
    # print('std for nonpara func:%.6f'%(std_non))
    # print('rmse for nonpara func:%.6f'%(rmse_non))
    print('rmse for solution u:%.3fe-2'%(rmse_u*1e2))

if model == 'baseline1':
    bias = np.mean(para_mat,axis=1) - 3e-3
    std = np.sqrt(np.mean(np.square(para_mat - np.mean(para_mat,axis=1)),axis=1))
    rmse = np.sqrt(bias**2 + std**2)
    # bias_non = np.sqrt(np.mean(np.square(np.mean(nonpara_mat,axis=1))))
    # std_non = np.sqrt(np.mean(np.square(nonpara_mat -np.mean(nonpara_mat,axis=1,keepdims=True))))
    # rmse_non = np.sqrt(bias_non**2 + std_non**2)
    rmse_u = np.sqrt(np.mean(u_mat,axis=1))
    print('For baseline1:')
    print('bias for para:%.3fe-2'%(bias*1e2))
    print('std for para:%.3fe-2'%(std*1e2))
    print('rmse for para:%.3fe-2'%(rmse*1e2))
    # print('bias for nonpara func:%.6f'%(bias_non))
    # print('std for nonpara func:%.6f'%(std_non))
    # print('rmse for nonpara func:%.6f'%(rmse_non))
    print('rmse for solution u:%.3fe-2'%(rmse_u*1e2))

if model == 'baseline2':
    # bias = np.mean(para_mat,axis=1) - 3e-3
    # std = np.sqrt(np.mean(np.square(para_mat - np.mean(para_mat,axis=1)),axis=1))
    # rmse = np.sqrt(bias**2 + std**2)
    # bias_non = np.sqrt(np.mean(np.square(np.mean(nonpara_mat,axis=1))))
    # std_non = np.sqrt(np.mean(np.square(nonpara_mat -np.mean(nonpara_mat,axis=1,keepdims=True))))
    # rmse_non = np.sqrt(bias_non**2 + std_non**2)
    rmse_u = np.sqrt(np.mean(u_mat,axis=1))
    print('For baseline2:')
    # print('bias for para:%.6f'%(bias))
    # print('std for para:%.6f'%(std))
    # print('rmse for para:%.6f'%(rmse))
    # print('bias for nonpara func:%.6f'%(bias_non))
    # print('std for nonpara func:%.6f'%(std_non))
    # print('rmse for nonpara func:%.6f'%(rmse_non))
    print('rmse for solution u:%.3fe-2'%(rmse_u*1e2))