import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import os
import scipy.io


import realdata_2_model
import realdata_2_dataset

if not os.path.isdir('pic'):
    os.mkdir('pic')
    
depth_name = 'h33'
case_name = 'VD3'
if case_name == 'VD2':
    h_v = 139
else:
    h_v = 556
b_v = 0.01
dim = 2
hidden=[16,64,64,16,1]

currents_name = ['Following_cw03']
mat = scipy.io.loadmat('WRR_Flume.mat')


check0 = torch.load('./checkpoint/realdata_2_para_'+depth_name+case_name+'0.pth')
check1 = torch.load('./checkpoint/realdata_2_para_'+depth_name+case_name+'1.pth')
check2 = torch.load('./checkpoint/realdata_2_para_'+depth_name+case_name+'2.pth')


# C_d_baseline_1 = np.ndarray((1,3))
# # currents_name = ['Purewave','Following_cw03','Following_cw06','Following_cw09','Following_cw12',
# #                  'Following_cw15','Opposing_cw03','Opposing_cw06']
# repreat_name = ['Cd_fit_01','Cd_fit_02','Cd_fit_03']
# for i in range(1):
#     for j in range(3):
#         C_d_baseline_1[i,j] = mat['WRR_Flume'][0,0]['WRR_Flume_2019']['h20'][0,0]['VD2'][0,0][currents_name[i]][0,0]['wave05cm08s'][0,0]['Cd'][0,0][repreat_name[j]]
# print("C_D 1:",check1['C_D'])
# print('C_D 1 baseline:',np.mean(C_d_baseline_1))

# C_d_baseline_2 = np.ndarray((1,3))
# # currents_name = ['Purewave','Following_cw03','Following_cw06','Opposing_cw03','Opposing_cw06']
# repreat_name = ['Cd_fit_01','Cd_fit_02','Cd_fit_03']
# for i in range(1):
#     for j in range(3):
#         C_d_baseline_2[i,j] = mat['WRR_Flume'][0,0]['WRR_Flume_2019']['h20'][0,0]['VD3'][0,0][currents_name[i]][0,0]['wave05cm08s'][0,0]['Cd'][0,0][repreat_name[j]]
# print("C_D 2:",check2['C_D'])
# print('C_D 2 baseline:',np.mean(C_d_baseline_2))

# C_d_baseline_3 = np.ndarray((1,3))
# # currents_name = ['Purewave','Following_cw03','Following_cw06','Following_cw09','Opposing_cw03','Opposing_cw06']
# repreat_name = ['Cd_fit_01','Cd_fit_02','Cd_fit_03']
# for i in range(1):
#     for j in range(3):
#         C_d_baseline_3[i,j] = mat['WRR_Flume'][0,0]['WRR_Flume_2019']['h33'][0,0]['VD2'][0,0][currents_name[i]][0,0]['wave05cm08s'][0,0]['Cd'][0,0][repreat_name[j]]
# print("C_D 3:",check3['C_D'])
# print('C_D 3 baseline:',np.mean(C_d_baseline_3))

C_d_baseline = np.ndarray((1,3))
C_d_baseline2 = np.ndarray((1,3))
# currents_name = ['Purewave','Following_cw03','Following_cw06','Following_cw09','Opposing_cw03','Opposing_cw06']
repeat_name = ['Cd_fit_01','Cd_fit_02','Cd_fit_03']
repeat_name2 = ['Cd_dir_01','Cd_dir_02','Cd_dir_03']
for i in range(1):
    for j in range(3):
        C_d_baseline[i,j] = mat['WRR_Flume'][0,0]['WRR_Flume_2019'][depth_name][0,0][case_name][0,0][currents_name[i]][0,0]['wave03cm08s'][0,0]['Cd'][0,0][repeat_name[j]]
        C_d_baseline2[i,j] = mat['WRR_Flume'][0,0]['WRR_Flume_2019'][depth_name][0,0][case_name][0,0][currents_name[i]][0,0]['wave03cm08s'][0,0]['Cd'][0,0][repeat_name2[j]][0,0][0,j]
print("C_D:",(check0['C_D']+check1['C_D']+check2['C_D'])/3)
# print(check0['C_D'],check1['C_D'],check2['C_D'])
print('C_D baseline:',np.mean(C_d_baseline))
print('C_D baseline2:',np.mean(C_d_baseline2))
print('C_D baseline3:',mat['WRR_Flume'][0,0]['WRR_Flume_2019'][depth_name][0,0][case_name][0,0][currents_name[i]][0,0]['wave03cm08s'][0,0]['Cd'][0,0]['Cd_cal'])


# t_ =  np.linspace(0,9.6,481,endpoint=True)
# if depth_name == 'h20':
#     if case_name == 'VD2':
#         dx0 = 0.8/27
#         curve_para0 = {'a':0.06251111, 'b':2*torch.pi*-0.099, 'c': 0.02524663,
#                         'position':torch.tensor([x*dx0 - dx0*14 for x in range(29)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*14 for x in range(29)]).float()/4.8}
#         position0 = [1,14,27]
#         # dx1 = 0.8/37
#         # curve_para1 = {'a':0.06785497, 'b':2*torch.pi*-0.2046, 'c': 0.02594901,
#         #                 'position':torch.tensor([x*dx1 - dx1*17 for x in range(39)]).float(),
#         #                 'init_phase':2*torch.pi*torch.tensor([x*0.187 - 0.187*17 for x in range(39)]).float()/4.8}
#         # position1 = [1,17,37] 
#         dx1 = 0.8/45
#         curve_para1 = {'a':0.06785497, 'b':2*torch.pi*-0.19, 'c': 0.02594901,
#                         'position':torch.tensor([x*dx1 - dx1*21 for x in range(47)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.154 - 0.154*21 for x in range(47)]).float()/4.8}
#         position1 = [1,21,45] 
#         dx2 = 0.8/35
#         curve_para2 = {'a':0.06669705, 'b':2*torch.pi*-0.08, 'c': 0.02702422,
#                         'position':torch.tensor([x*dx2 - dx2*9 for x in range(37)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*9 for x in range(37)]).float()/4.8}
#         position2 = [1,9,35] 
#     else:
#         dx0 = 0.8/21
#         curve_para0 = {'a':0.04986418, 'b':0, 'c': 0.02830579,
#                         'position':torch.tensor([x*dx0 - dx0*13 for x in range(23)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*13 for x in range(23)]).float()/4.8}
#         position0 = [1,13,21]
#         dx1 = 0.8/36
#         curve_para1 = {'a':0.04784506, 'b': 2*torch.pi*-0.2, 'c': 0.02701595,
#                         'position':torch.tensor([x*dx1 - dx1*11 for x in range(38)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*11 for x in range(38)]).float()/4.8}
#         position1 = [1,11,36] 
#         dx2 = 0.8/48
#         curve_para2 = {'a':0.0484061, 'b':2*torch.pi*0.48, 'c': 0.02693719,
#                         'position':torch.tensor([x*dx2 - dx2*20 for x in range(50)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*20 for x in range(50)]).float()/4.8}
#         position2 = [1,20,48] 
# else:
#     if case_name == 'VD2':
#         dx0 = 0.8/44
#         curve_para0 = {'a':0.04204181, 'b':2*torch.pi*0.71, 'c': 0.0189894,
#                         'position':torch.tensor([x*dx0 -dx0*24 for x in range(46)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.15 - 0.15*24 for x in range(46)]).float()/4.8}
#         position0 = [1,24,44]
#         dx1 = 0.8/32
#         curve_para1 = {'a':0.04169234, 'b':2*torch.pi*0.25, 'c': 0.02054691,
#                         'position':torch.tensor([x*dx1 - dx1*16 for x in range(34)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*16 for x in range(34)]).float()/4.8}
#         position1 = [1,16,32]
#         dx2 = 0.8/34
#         curve_para2 = {'a':0.04352463, 'b':2*torch.pi*-0.14, 'c': 0.02208982,
#                         'position':torch.tensor([x*dx2 - dx2*14 for x in range(36)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*14 for x in range(36)]).float()/4.8}
#         position2 = [1,14,34] 
#     else:
#         dx0 = 0.8/30
#         curve_para0 = {'a':0.03674994, 'b':2*torch.pi*0.015/0.48, 'c': 0.01470599,
#                         'position':torch.tensor([x*dx0 - dx0*17 for x in range(32)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*17 for x in range(32)]).float()/4.8}
#         position0 = [1,17,30]
#         dx1 = 0.8/28
#         curve_para1 = {'a':0.03756187, 'b':2*torch.pi*-0.11, 'c': 0.01290169,
#                         'position':torch.tensor([x*dx1 - dx1*18 for x in range(30)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*18 for x in range(30)]).float()/4.8}
#         position1 = [1,18,28]
#         dx2 = 0.8/20
#         curve_para2 = {'a':0.03550256, 'b':2*torch.pi*1.35, 'c': 0.01259831,
#                         'position':torch.tensor([x*dx2 - dx2*17 for x in range(22)]).float(),
#                         'init_phase':2*torch.pi*torch.tensor([x*0.24375 - 0.24375*17 for x in range(22)]).float()/4.8}
#         position2 = [1,17,20]

        
# model_dic_0 = check0['net']
# para_model = realdata_2_model.ParametricPart(h_v,b_v,dx=dx0)
# nonpara_model = realdata_2_model.NonParametricModel(state_dim=dim,hidden=hidden)
# mode = 1
# penalty = None
# full_model = realdata_2_model.SemiParametericModel(para_model,nonpara_model,h_v,b_v,mode=mode,penalty=penalty)
# full_model.nonpara_model.net.load_state_dict(model_dic_0)
# full_model.para_model._params['C_D'] = torch.Tensor.detach(torch.Tensor.cpu(check0['C_D']))
# full_model.para_model._params['nv'] = torch.Tensor.detach(torch.Tensor.cpu(check0['nv']))
# data_0 = realdata_2_dataset.Datagen("./data/testdata_"+depth_name+case_name+".npy",curve_para0,0)
# test_0 = data_0.__getitem__(0)
# full_model.nonpara_model.change_x_position(test_0['x_position'].reshape(1,-1))
# pred_0 = torch.Tensor.detach(full_model(test_0['initial'].reshape(1,-1),test_0['t']))[0,:,position0].numpy()
# con_0 = np.ndarray((481,3))
# true_0 = np.load("./data/testdata_"+depth_name+case_name+".npy")[0,0,:,:]
# for i in range(6):
#     con_0[(i*80):(i*80+81),:] = pred_0
    
# fig = plt.figure()
# ax = fig.add_subplot(911)
# ax.plot(t_,con_0[:,0],'--',t_,true_0[:,0],'-')
# ax.set_ylabel('E1P1')
# ax.xaxis.set_ticks([])
# ax = fig.add_subplot(912)
# ax.plot(t_,con_0[:,1],'--',t_,true_0[:,1],'-')
# ax.set_ylabel('E1P2')
# ax.xaxis.set_ticks([])
# ax = fig.add_subplot(913)
# ax.plot(t_,con_0[:,2],'--',t_,true_0[:,2],'-')
# ax.set_ylabel('E1P3')
# ax.xaxis.set_ticks([])

# model_dic_1 = check1['net']
# para_model = realdata_2_model.ParametricPart(h_v,b_v,dx=dx1)
# nonpara_model = realdata_2_model.NonParametricModel(state_dim=dim,hidden=hidden)
# mode = 1
# penalty = None
# full_model = realdata_2_model.SemiParametericModel(para_model,nonpara_model,h_v,b_v,mode=mode,penalty=penalty)
# full_model.nonpara_model.net.load_state_dict(model_dic_1)
# full_model.para_model._params['C_D'] = torch.Tensor.detach(torch.Tensor.cpu(check1['C_D']))
# full_model.para_model._params['nv'] = torch.Tensor.detach(torch.Tensor.cpu(check1['nv']))
# data_1 = realdata_2_dataset.Datagen("./data/testdata_"+depth_name+case_name+".npy",curve_para1,1)
# test_1 = data_1.__getitem__(0)
# full_model.nonpara_model.change_x_position(test_1['x_position'].reshape(1,-1))
# pred_1 = torch.Tensor.detach(full_model(test_1['initial'].reshape(1,-1),test_1['t']))[0,:,position1].numpy()
# con_1 = np.ndarray((481,3))
# true_1 = np.load("./data/testdata_"+depth_name+case_name+".npy")[0,1,:,:]
# for i in range(6):
#     con_1[(i*80):(i*80+81),:] = pred_1
# ax = fig.add_subplot(914)
# ax.plot(t_,con_1[:,0],'--',t_,true_1[:,0],'-')
# ax.set_ylabel('E2P1')
# ax.xaxis.set_ticks([])
# ax = fig.add_subplot(915)
# ax.plot(t_,con_1[:,1],'--',t_,true_1[:,1],'-')
# ax.set_ylabel('E2P2')
# ax.xaxis.set_ticks([])
# ax = fig.add_subplot(916)
# ax.plot(t_,con_1[:,2],'--',t_,true_1[:,2],'-')
# ax.set_ylabel('E2P3')
# ax.xaxis.set_ticks([])


# model_dic_2 = check2['net']
# para_model = realdata_2_model.ParametricPart(h_v,b_v,dx=dx2)
# nonpara_model = realdata_2_model.NonParametricModel(state_dim=dim,hidden=hidden)
# mode = 1
# penalty = None
# full_model = realdata_2_model.SemiParametericModel(para_model,nonpara_model,h_v,b_v,mode=mode,penalty=penalty)
# full_model.nonpara_model.net.load_state_dict(model_dic_2)
# full_model.para_model._params['C_D'] = torch.Tensor.detach(torch.Tensor.cpu(check2['C_D']))
# full_model.para_model._params['nv'] = torch.Tensor.detach(torch.Tensor.cpu(check2['nv']))
# data_2 = realdata_2_dataset.Datagen("./data/testdata_"+depth_name+case_name+".npy",curve_para2,0)
# test_2 = data_2.__getitem__(0)
# full_model.nonpara_model.change_x_position(test_2['x_position'].reshape(1,-1))
# pred_2 = torch.Tensor.detach(full_model(test_2['initial'].reshape(1,-1),test_2['t']))[0,:,position2].numpy()
# con_2 = np.ndarray((481,3))
# true_2 = np.load("./data/testdata_"+depth_name+case_name+".npy")[0,2,:,:]
# for i in range(6):
#     con_2[(i*80):(i*80+81),:] = pred_2
# ax = fig.add_subplot(917)
# ax.plot(t_,con_2[:,0],'--',t_,true_2[:,0],'-')
# ax.set_ylabel('E3P1')
# ax.xaxis.set_ticks([])
# ax = fig.add_subplot(918)
# ax.plot(t_,con_2[:,1],'--',t_,true_2[:,1],'-')
# ax.set_ylabel('E3P2')
# ax.xaxis.set_ticks([])
# ax = fig.add_subplot(919)
# ax.plot(t_,con_2[:,2],'--',t_,true_2[:,2],'-')
# ax.set_ylabel('E3P3')
# # ax.xaxis.set_ticks([])
# fig.text(0.5, 0.025, 'time (s)', ha='center',fontsize=16)
# fig.text(0.01, 0.5, 'velocity (m/s)', va='center', rotation='vertical',fontsize=16)

# plt.savefig('./pic/curve_'+depth_name+case_name+'.png')

