import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import os

import realdata_1_model

# mode
para_plot = 0
non_part = 1

data_1 = np.load('./outputs/realdata_1.npy')
data_1_modify = np.load('./outputs/realdata_1_modify.npy')
data_1_baseline = np.load('./outputs/realdata_1_paramodel.npy')
data_1_baseline_modify = np.load('./outputs/realdata_1_paramodel_modify.npy')
leng = data_1.shape[1]
epoch_list = np.linspace(0,leng*4,leng,endpoint=False) + 4
if not os.path.isdir('pic'):
    os.mkdir('pic')
    
    
    
if para_plot:    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(epoch_list,data_1[2,:leng]*1000,'r-',epoch_list,data_1_modify[2,:leng]*1000,'b-')
    # ax.legend(labels = ['Model 1','Model 2'],loc=(0.72,0.45))
    # ax2 = ax.twinx()
    # ax2.plot(epoch_list,data_1[1,:leng],'r-',epoch_list,data_1_modify[1,:leng],'b-')
    # # ax.grid()
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel('Value of parameter D')
    # ax2.set_ylabel('Test loss')
    # plt.savefig("./pic/para_pic.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epoch_list,data_1[1,:leng],epoch_list,data_1_baseline[1,:leng],epoch_list,data_1_modify[1,:leng],epoch_list,data_1_baseline_modify[1,:leng])
    ax.set_xlabel("Epoch",fontsize=16)
    ax.set_ylabel('Test loss',fontsize=16)
    ax.legend(labels = ['semiPDE 1','Benchmark 1','semiPDE 2','Benchmark 2'],loc=(0.72,0.75))
    axins = ax.inset_axes((0.25, 0.3, 0.48, 0.36))
    axins.plot(epoch_list,data_1[1,:leng],epoch_list,data_1_baseline[1,:leng],epoch_list,data_1_modify[1,:leng],epoch_list,data_1_baseline_modify[1,:leng])
    axins.set_xlim(800,1000)
    axins.set_ylim(1.5e-8,2.25e-8)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.savefig("./pic/loss_pic.png")

if non_part:
    dim = 1
    hidden=[16,64,64,16,dim]
    K = 1.7e-3
    # lamda = 0.055
    check_1 = torch.load('./checkpoint/realdata_1.pth')
    check_2 = torch.load('./checkpoint/realdata_1_modify.pth')
    check_3 = torch.load('./checkpoint/realdata_1_paramodel.pth')
    check_4 = torch.load('./checkpoint/realdata_1_paramodel_modify.pth')
    print("D1:",check_1['D'])
    print("D2:",check_2['D'])
    print("D3:",check_3['D'])
    print("D4:",check_4['D'])
    print("loss1:",check_1['test_loss'])
    print("loss2:",check_2['test_loss'])
    print("loss3:",check_3['test_loss'])
    print("loss4:",check_4['test_loss'])
    net_mat = check_1['net']
    net_mat_modify = check_2['net']
    x = np.linspace(0.0002,0.0016,100) 
    x_ = torch.tensor(x.reshape(-1,1)).float()
    
    f = realdata_1_model.NonParametricModel(state_dim=dim,hidden=hidden)
    f.net.load_state_dict(net_mat)
    y_ = f(x_*1000 - 1)
    y = torch.Tensor.detach(y_).numpy().reshape(-1,1)*0.00001
    
    f.net.load_state_dict(net_mat_modify)
    y_ = f(x_*1000 - 1)
    y_modify = torch.Tensor.detach(y_).numpy().reshape(-1,1)*0.00001
    
    lamda = torch.Tensor.detach(torch.Tensor.cpu(check_3['lamda'])).numpy()
    print('lamda1:',lamda)
    z = lamda * x * (1-x/K)
    
    lamda_modify = torch.Tensor.detach(torch.Tensor.cpu(check_4['lamda'])).numpy()
    print('lamda1:',lamda_modify)
    z_modify = lamda_modify * x * (1-x/K)
    plt.plot(x,y,'r-',x,z,'r--',x,y_modify,'b-',x,z_modify,'b--')
    plt.legend(labels = ['semiPDE 1','Benchmark 1','semiPDE 2','Benchmark 2'],loc=(0.72,0.75))
    # plt.plot(x,y,'r-',x,z,'b--')
    # plt.legend(labels = ['semiPDE 1','Benchmark 1'],loc=(0.72,0.75))
    #plt.ylim(0,2.5)
    plt.ylabel('reaction value f(u)',fontsize=16)
    plt.xlabel('density u',fontsize=16)
    plt.savefig("./pic/nonpart_pic.png")