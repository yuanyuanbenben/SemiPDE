import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import RD_model


dim=1
hidden=[16,64,64,16,dim]
test_x = np.linspace(0,1,1001,endpoint=True).reshape(1001,1)
test_x = torch.tensor(test_x).float()
true_f = lambda u: u*(1-u)
para_mat = np.ndarray((1,500))
nonpara_mat = np.ndarray((1001,500))
u_mat = np.ndarray((1,500))
sigma_list = np.ndarray((1,500))
size1 = 0
size2 = 0
size3 = 0
ts = 10
ss = 16
# epo = np.ndarray((100))
for i in range(500):
    checkpoints = torch.load('checkpoint_para/ckpt_rd_64_16_%d_0.100_0.003_%d_%d.pth'%(ts,ss,i))
    model_net = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
    model_net.net.load_state_dict(checkpoints['net'])
    para_mat[:,i] = checkpoints['D'].to('cpu').detach().numpy()*1e-3
    # nonpara_mat[:,i:(i+1)] = (model_net.net(test_x) - true_f(test_x)).detach().numpy()
    # u_mat[:,i] = checkpoints['u_loss']
    checkpoints_2 = torch.load('checkpoint_para/ckpt_rd_variance_64_16_%d_0.100_0.003_%d_%d.pth'%(ts,ss,i))
    sigma_list[:,i] = np.sqrt(checkpoints['test_loss'] / 64 / ts/ ss / checkpoints_2['test_loss'])
    if np.abs(para_mat[:,i] - 3e-3) > 1.96 * sigma_list[:,i]:
        size1 = size1 + 1
    if np.abs(para_mat[:,i] - 3e-3) > 1.645 * sigma_list[:,i]:
        size2 = size2 + 1
    if np.abs(para_mat[:,i] - 3e-3) > 1.28 * sigma_list[:,i]:
        size3 = size3 + 1
bias = np.mean(para_mat,axis=1) - 3e-3
std = np.sqrt(np.mean(np.square(para_mat - np.mean(para_mat,axis=1)),axis=1))
rmse = np.sqrt(bias**2 + std**2)
bias_non = np.sqrt(np.mean(np.square(np.mean(nonpara_mat,axis=1))))
std_non = np.sqrt(np.mean(np.square(nonpara_mat -np.mean(nonpara_mat,axis=1,keepdims=True))))
rmse_non = np.sqrt(bias_non**2 + std_non**2)
rmse_u = np.sqrt(np.mean(u_mat,axis=1))
std_sigma = np.mean(sigma_list)
print('bias for para:%.3fe-5'%(bias*1e4))
print('std for para:%.3fe-4'%(std*1e4))
print('rmse for para:%.3fe-4'%(rmse*1e4))
print('sigma hat:%.3fe-4'%(std_sigma*1e4))
print('coverage rate for 0.95 confidence interval:', 1 - size1/500)
print('coverage rate for 0.90 confidence interval:', 1 - size2/500)
print('coverage rate for 0.80 confidence interval:', 1 - size3/500)
