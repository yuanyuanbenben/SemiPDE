# real data experiment based on the laboratory dataset on wave propagation through vegetation
import numpy as np
import scipy.io
import torch
import os

import realdata_2_dataset
import realdata_2_model
import realdata_2_optimal


# setting parameters
train_batch_size = 2 #2/6
test_batch_size = 1 #1/3
tau_1 = 1e-3
niter = 5
nupdate = 36
nepoch = 10000
nlog = 9
lambda_0 = 1
tau_2 = 1
dim=1
hidden=[16,64,64,16,dim]

# laboratory dataset in 2019 is pre-processed as train and test dataset
# h25 and h50 are the depth of water
# depth_name = ['h20','h33']
# VD0 is the the control test with mimic term density N = 0, VD2,VD3 with N = 139,556
# case_name = ['VD2','VD3']
# currents
# currents_name = ['Purewave','Following_cw03','Following_cw06','Following_cw09','Following_cw12',
#                     'Following_cw15','Opposing_cw03','Opposing_cw06','Opposing_cw09','Opposing_cw12','Opposing_cw15']
# wave type 
# wave_name = ['wave03cm06s', 'wave03cm08s', 'wave05cm06s', 'wave05cm08s', 'wave05cm10s','wave07cm08s','wave07cm10s']

# select depth = 0.2m, density = 139 and wave number 05cm08s
depth_name = 'h33'
case_name = 'VD3'
repeat_curve = 2
# if depth_name == 'h20':
#     depth_num = [0.2,0.33][0]
#     if case_name == 'VD2':
#         currents_name = ['Following_cw06']
#     else:
#         currents_name = ['Following_cw06']
# else:
#     if case_name == 'VD2':
#         currents_name = ['Following_cw06']
#     else:
#         currents_name = ['Following_cw06']
currents_name = 'Following_cw03'
wave_name = 'wave03cm08s' 
repeat_u_name = ['U_01','U_02','U_03']
repeat_f_name = ['F_01','F_02','F_03']
if case_name == 'VD2':
    h_v = 139
else:
    h_v = 556
b_v = 0.01

            
traindata_path = "./data/traindata_" + depth_name + case_name + ".npy"
testdata_path = "./data/testdata_" + depth_name + case_name + ".npy"
path_save = "./outputs/realdata_2_" + depth_name + case_name + str(repeat_curve)+".npy"
path_checkpoint = "./checkpoint/realdata_2_" + depth_name + case_name + str(repeat_curve)+".pth"


device = torch.device('cuda:1')

if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')    

# check whether original dataset has been saved as numpy form
if (not os.path.exists(traindata_path)) or (not os.path.exists(testdata_path)):
    # import data
    mat = scipy.io.loadmat('WRR_Flume.mat')
    mat = mat['WRR_Flume'][0,0]['WRR_Flume_2019'][depth_name][0,0][case_name][0,0]
    # current * repeat * time * x_space
    # if depth_name == 'h20':
    #     if case_name == 'VD2':
    #         total_data = np.ndarray((1,3,481,3))
    #     else:
    #         total_data = np.ndarray((1,3,481,3))
    # else:
    #     if case_name == 'VD2':
    #         total_data = np.ndarray((1,3,481,3))
    #     else:
    #         total_data = np.ndarray((1,3,481,3))
    total_data = np.ndarray((1,3,481,3))
    for currents in range(len(currents_name)):
        for repeat in range(len(repeat_u_name)):
            total_data[currents,repeat,:,:] = mat[currents_name[currents]][0,0][wave_name][0,0]['U'][0,0][repeat_u_name[repeat]][0,0]
            #total_data[currents,wave,repeat,1,:,:] = mat[currents_name[currents]][0,0][wave_name[wave]][0,0]['F'][0,0][repeat_f_name[repeat]][0,0]
    # if depth_name == 'h20':
    #     if case_name == 'VD2':      
    #         train_data = total_data[:,[0,1],:,:]
    #         test_data = total_data[:,2:3,:,:]
    #     else:
    #         train_data = total_data[:,[0,1],:,:]
    #         test_data = total_data[:,2:3,:,:]
    # else:
    #     if case_name == 'VD2':
    #         train_data = total_data[:,[0,1],:,:]
    #         test_data = total_data[:,2:3,:,:]
    #     else:
    #         train_data = total_data[:,[0,1],:,:]
    #         test_data = total_data[:,2:3,:,:]
    np.save(traindata_path,total_data)
    np.save(testdata_path,total_data)
    
# generate training and testing dataflow
print('==> Preparing data...')
# initial state
# a: Uw_01 in original data
# c: Umean_01 in original data
# b: phase


# for 'wave03cm08s' and 'Following_cw03'
if depth_name == 'h20':
    if case_name == 'VD2':
        if repeat_curve == 0:
            dx = 0.8/27
            curve_para = {'a':0.06251111, 'b':2*torch.pi*-0.099, 'c': 0.02524663,
                        'position':torch.tensor([x*dx - dx*14 for x in range(29)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*14 for x in range(29)]).float()/4.8}
            position = [1,14,27]
        elif repeat_curve == 1:
            dx = 0.8/45
            curve_para = {'a':0.06785497, 'b':2*torch.pi*-0.19, 'c': 0.02594901,
                        'position':torch.tensor([x*dx - dx*21 for x in range(47)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.154 - 0.154*21 for x in range(47)]).float()/4.8}
            position = [1,21,45] 
        else:
            dx = 0.8/35
            curve_para = {'a':0.06669705, 'b':2*torch.pi*-0.08, 'c': 0.02702422,
                        'position':torch.tensor([x*dx - dx*9 for x in range(37)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2- 0.2*9 for x in range(37)]).float()/4.8}
            position = [1,9,35] 
        nepoch = 15000
    else:
        if repeat_curve == 0:
            dx = 0.8/21
            curve_para = {'a':0.04986418, 'b':0, 'c': 0.02830579,
                        'position':torch.tensor([x*dx - dx*13 for x in range(23)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*13 for x in range(23)]).float()/4.8}
            position = [1,13,21]
        elif repeat_curve == 1:
            dx = 0.8/36
            curve_para = {'a':0.04784506, 'b': 2*torch.pi*-0.2, 'c': 0.02701595,
                        'position':torch.tensor([x*dx - dx*11 for x in range(38)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*11 for x in range(38)]).float()/4.8}
            position = [1,11,36] 
        else:
            dx = 0.8/48
            curve_para = {'a':0.0484061, 'b':2*torch.pi*0.48, 'c': 0.02693719,
                        'position':torch.tensor([x*dx - dx*20 for x in range(50)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*20 for x in range(50)]).float()/4.8}
            position = [1,20,48] 
else:
    if case_name == 'VD2':
        if repeat_curve == 0:
            dx = 0.8/44
            curve_para = {'a':0.04204181, 'b':2*torch.pi*0.71, 'c': 0.0189894,
                        'position':torch.tensor([x*dx -dx*24 for x in range(46)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.15 - 0.15*24 for x in range(46)]).float()/4.8}
            position = [1,24,44]
        elif repeat_curve == 1:
            dx = 0.8/32
            curve_para = {'a':0.04169234, 'b':2*torch.pi*0.25, 'c': 0.02054691,
                        'position':torch.tensor([x*dx - dx*16 for x in range(34)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*16 for x in range(34)]).float()/4.8}
            position = [1,16,32] 
        else:
            dx = 0.8/34
            curve_para = {'a':0.04352463, 'b':2*torch.pi*-0.14, 'c': 0.02208982,
                        'position':torch.tensor([x*dx - dx*14 for x in range(36)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*14 for x in range(36)]).float()/4.8}
            position = [1,14,34] 
    else:
        if repeat_curve == 0:
            dx = 0.8/30
            curve_para = {'a':0.03674994, 'b':2*torch.pi*0.015/0.48, 'c': 0.01470599,
                        'position':torch.tensor([x*dx - dx*17 for x in range(32)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*17 for x in range(32)]).float()/4.8}
            position = [1,17,30]
        elif repeat_curve == 1:
            dx = 0.8/28
            curve_para = {'a':0.03756187, 'b':2*torch.pi*-0.11, 'c': 0.01290169,
                        'position':torch.tensor([x*dx - dx*18 for x in range(30)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*18 for x in range(30)]).float()/4.8}
            position = [1,18,28]
        else:
            dx = 0.8/20
            curve_para = {'a':0.03550256, 'b':2*torch.pi*1.35, 'c': 0.01259831,
                        'position':torch.tensor([x*dx - dx*17 for x in range(22)]).float(),
                        'init_phase':2*torch.pi*torch.tensor([x*0.24375 - 0.24375*17 for x in range(22)]).float()/4.8}
            position = [1,17,20]

# # for 'wave03cm08s' and 'Opposing_cw03'
# if depth_name == 'h20':
#     if case_name == 'VD2':
#         if repeat_curve == 0:
#             dx = 0.8/34
#             curve_para = {'a':0.06307863, 'b':2*torch.pi*0.29, 'c': -0.03102722,
#                           'position':torch.tensor([x*dx - dx*18 for x in range(36)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*18 for x in range(36)]).float()/4.8}
#             position = [1,18,34]
#         elif repeat_curve == 1:
#             dx = 0.8/27
#             curve_para = {'a':0.05817452, 'b':2*torch.pi*0.415, 'c': -0.03197422,
#                           'position':torch.tensor([x*dx - dx*17 for x in range(29)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.1 - 0.1*17 for x in range(29)]).float()/4.8}
#             position = [1,17,27] 
#         else:
#             dx = 0.8/22
#             curve_para = {'a':0.06136102, 'b':2*torch.pi*0.3, 'c': -0.03127445,
#                           'position':torch.tensor([x*dx - dx*12 for x in range(24)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.1 - 0.1*12 for x in range(24)]).float()/4.8}
#             position = [1,12,22] 
#     else:
#         if repeat_curve == 0:
#             dx = 0.8/44
#             curve_para = {'a':0.04069125, 'b':2*torch.pi*-0.45, 'c': -0.02175844,
#                           'position':torch.tensor([x*dx - dx*23 for x in range(46)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*23 for x in range(46)]).float()/4.8}
#             position = [1,23,44]
#         elif repeat_curve == 1:
#             dx = 0.8/45
#             curve_para = {'a':0.04184063, 'b': 2*torch.pi*-0.16, 'c': -0.02182146,
#                           'position':torch.tensor([x*dx - dx*25 for x in range(47)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*25 for x in range(47)]).float()/4.8}
#             position = [1,25,45] 
#         else:
#             dx = 0.8/44
#             curve_para = {'a':0.03726876, 'b':2*torch.pi*-0.4, 'c': -0.02358073,
#                           'position':torch.tensor([x*dx - dx*25 for x in range(46)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*25 for x in range(46)]).float()/4.8}
#             position = [1,25,44] 
# else:
#     if case_name == 'VD2':
#         if repeat_curve == 0:
#             dx = 0.8/44
#             curve_para = {'a':0.04204181, 'b':2*torch.pi*0.71, 'c': 0.0189894,
#                           'position':torch.tensor([x*dx -dx*24 for x in range(46)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.15 - 0.15*24 for x in range(46)]).float()/4.8}
#             position = [1,24,44]
#         elif repeat_curve == 1:
#             dx = 0.8/32
#             curve_para = {'a':0.04169234, 'b':2*torch.pi*0.25, 'c': 0.02054691,
#                           'position':torch.tensor([x*dx - dx*16 for x in range(34)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*16 for x in range(34)]).float()/4.8}
#             position = [1,16,32] 
#         else:
#             dx = 0.8/34
#             curve_para = {'a':0.04352463, 'b':2*torch.pi*-0.14, 'c': 0.02208982,
#                           'position':torch.tensor([x*dx - dx*14 for x in range(36)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*14 for x in range(36)]).float()/4.8}
#             position = [1,14,34] 
#     else:
#         if repeat_curve == 0:
#             dx = 0.8/30
#             curve_para = {'a':0.03674994, 'b':2*torch.pi*0.015/0.48, 'c': 0.01470599,
#                           'position':torch.tensor([x*dx - dx*17 for x in range(32)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*17 for x in range(32)]).float()/4.8}
#             position = [1,17,30]
#         elif repeat_curve == 1:
#             dx = 0.8/28
#             curve_para = {'a':0.03756187, 'b':2*torch.pi*-0.11, 'c': 0.01290169,
#                           'position':torch.tensor([x*dx - dx*18 for x in range(30)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.2 - 0.2*18 for x in range(30)]).float()/4.8}
#             position = [1,18,28]
#         else:
#             dx = 0.8/20
#             curve_para = {'a':0.03550256, 'b':2*torch.pi*1.35, 'c': 0.01259831,
#                           'position':torch.tensor([x*dx - dx*17 for x in range(22)]).float(),
#                           'init_phase':2*torch.pi*torch.tensor([x*0.24375 - 0.24375*17 for x in range(22)]).float()/4.8}
#             position = [1,17,20]
train_data = realdata_2_dataset.creat_data(traindata_path,train_batch_size,curve_para,repeat_curve)
test_data = realdata_2_dataset.creat_data(testdata_path,test_batch_size,curve_para,repeat_curve)


# construct model
print('==> Preparing model...')

para_model = realdata_2_model.ParametricPart(h_v,b_v,dx=dx)
nonpara_model = realdata_2_model.NonParametricModel(state_dim=2,hidden=hidden)

mode = None


if mode is None:
    print('using para_model')
    path_save = "./outputs/realdata_2_para_" + depth_name + case_name + str(repeat_curve)+".npy"
    path_checkpoint = "./checkpoint/realdata_2_para_" + depth_name + case_name + str(repeat_curve)+".pth"
    
penalty = None# 'orthogonal'
#whole model
full_model = realdata_2_model.SemiParametericModel(para_model,nonpara_model,h_v,b_v,mode=mode,penalty=penalty)
print('==> Model already !')



# optimal
print('==> Optimal start...')
optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
train_process = realdata_2_optimal.OptimalProcess(
    train=train_data,test=test_data,net=full_model,optimizer=optimizer,position=position,lambda_0=lambda_0,
    tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,
    path = path_checkpoint,device = device)
ret = train_process.run()
print('==> Optimal finish...')
train_loss = ret["train_loss_mat"]
test_loss = ret["test_loss_mat"]
parameter_value = ret["parameter_mat"]
parameter_value2 = ret["parameter_mat2"]
save_mat = np.array([train_loss,test_loss,parameter_value,parameter_value2])
np.save(path_save,save_mat)



    