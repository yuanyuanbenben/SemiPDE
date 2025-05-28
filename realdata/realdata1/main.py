# real data experiment for the reaction diffusion biology database
import numpy as np
import pandas as pd
import torch 
import os 

import realdata_1_dataset
import realdata_1_model
import realdata_1_optimal
import realdata_1_paramodel

# setting parameters
train_batch_size = 3
test_batch_size = 3
tau_1 = 1e-3
niter = 5
nupdate = 20
nepoch = 1000
nlog = 5
lambda_0 = 1
tau_2 = 1
range_grid = [0,1900]
num_grid = 38
dim = 1
hidden=[16,64,64,16,dim]

modify = False
true_model = True

traindata_path = "./data/traindata.npy"
testdata_path = "./data/testdata.npy"


if true_model:
    if modify:
        path_save = "./outputs/realdata_1_paramodel_modify.npy"
        path_checkpoint = "./checkpoint/realdata_1_paramodel_modify.pth"
    else:
        path_save = "./outputs/realdata_1_paramodel.npy"
        path_checkpoint = "./checkpoint/realdata_1_paramodel.pth"
else:
    if modify:
        path_save = "./outputs/realdata_1_modify.npy"
        path_checkpoint = "./checkpoint/realdata_1_modify.pth"
    else:
        path_save = "./outputs/realdata_1.npy"
        path_checkpoint = "./checkpoint/realdata_1.pth"
        
device = torch.device('cuda:0')

if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')    

# check whether original dataset has been saved as numpy form
# if not, create train and test dataset with 15 curves for training and 3 curves for testing
if (not os.path.exists(traindata_path)) or (not os.path.exists(testdata_path)):
    # import data
    total_data = np.ndarray((18,5,38))
    sheet_index =  ['0h','12h','24h','36h','48h']
    for case_num in range(6):
        for sheet_num in range(5):
            _data = pd.read_excel(f'1-s2.0-S0022519315005676-mmc{case_num+2}.xlsx',sheet_name=sheet_index[sheet_num],header=1,usecols=range(1,39))
            _data = _data.to_numpy()
            total_data[(case_num*3):(case_num*3+3),sheet_num,:] = _data[[2,4,6],:]
    train_data = total_data[[0,1,2,3,5,6,7,8,10,11,12,13,15,16,17],:,:]
    test_data = total_data[[4,9,14],:,:]
    np.save(traindata_path,train_data)
    np.save(testdata_path,test_data)
    
# generate training and testing dataflow
print('==> Preparing data...')
train_data = realdata_1_dataset.creat_data(traindata_path,train_batch_size,device)
test_data = realdata_1_dataset.creat_data(testdata_path,test_batch_size,device)

# construct model
print('==> Preparing model...')

if true_model:
    mode = None
    if modify:
        para_model = realdata_1_paramodel.TrueModifyParametricPart(dx=(range_grid[1]-range_grid[0])/num_grid)
    else:
        para_model = realdata_1_paramodel.TrueParametricPart(dx=(range_grid[1]-range_grid[0])/num_grid)
else:
    mode = 1
    if modify:
        para_model = realdata_1_model.ModifyParametricPart(dx=(range_grid[1]-range_grid[0])/num_grid)
    else:
        para_model = realdata_1_model.ParametricPart(dx=(range_grid[1]-range_grid[0])/num_grid)
        
nonpara_model = realdata_1_model.NonParametricModel(state_dim=dim,hidden=hidden)
penalty = None# 'orthogonal'
#whole model
full_model = realdata_1_model.SemiParametericModel(para_model,nonpara_model,mode=mode,penalty=penalty)
print('==> Model already !')


# optimal
print('==> Optimal start...')
optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
if true_model:
    train_process = realdata_1_paramodel.OptimalProcess(
        train=train_data,test=test_data,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device = device)
    ret = train_process.run()
    print('==> Optimal finish...')
    train_loss = ret["train_loss_mat"]
    test_loss = ret["test_loss_mat"]
    parameter_value_D = ret["parameter_mat_D"]
    parameter_value_lamda = ret["parameter_mat_lamda"]
    save_mat = np.array([train_loss,test_loss,parameter_value_D,parameter_value_lamda])
    np.save(path_save,save_mat)
else:
    train_process = realdata_1_optimal.OptimalProcess(
        train=train_data,test=test_data,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device = device)
    ret = train_process.run()
    print('==> Optimal finish...')
    train_loss = ret["train_loss_mat"]
    test_loss = ret["test_loss_mat"]
    parameter_value = ret["parameter_mat"]
    save_mat = np.array([train_loss,test_loss,parameter_value])
    np.save(path_save,save_mat)
