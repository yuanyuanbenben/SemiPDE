# main function
import torch
import os
import numpy as np

import RCD_compa_dataset
import RCD_compa_model
import RCD_compa_optimal 

# setting parameters

# data
D_1 = 1
D_2 = 0.1
func_f = None
bc = "periodic"
batch_size = [1,1]
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
noise_sigma=0.02 # 0.02/0.05
dim=1
# net
state_dim = 1
hidden=[16,64,64,16,dim]
lambda_0 = 1#5*64*25
tau_1 = 5e-3
tau_2 = 1
niter = 5
nupdate = 5
nepoch = 200
nlog= 1
dx = (range_grid[0][1]-range_grid[0][0])/num_grid[0]
path_train = "./data/rcd_train_%d_%d_%d_%.3f_%.3f_%.3f"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)
path_test = "./data/rcd_test_%d_%d_%d_%.3f_%.3f_%.3f"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)
path_save = "./outputs/rcd_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)

path_checkpoint = "./checkpoint/ckpt_rcd_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)
if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    
device = torch.device("cuda:0")

# generate data
print('==> Preparing data..')
train,test = RCD_compa_dataset.create_data(path_train=path_train,path_test=path_test,
    D_1=D_1,D_2=D_2,func_f=func_f,bc=bc,
    batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
    sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
    period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,noise_sigma=noise_sigma,dim=dim)
print('==> Data already !')

# construct model
print('==> Preparing model..')
para_model = RCD_compa_model.ParametricPart(dx=dx)
nonpara_model = RCD_compa_model.NonParametricModel(state_dim=state_dim,hidden=hidden,device=device)

mode = 1 # 1
if mode is None:
    path_save = "./outputs/rcd_incom_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)

    path_checkpoint = "./checkpoint/ckpt_rcd_incom_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)
penalty = None #'L_2'
if penalty is None:
    path_save = "./outputs/rcd_nopena_%d_%d_%d_%.3f_%.3f_%.3f.npy"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)

    path_checkpoint = "./checkpoint/ckpt_rcd_nopena_%d_%d_%d_%.3f_%.3f_%.3f.pth"%(train_curve_num,test_curve_num,
                                                                 sample_size,noise_sigma,D_1,D_2)
#whole model
full_model = RCD_compa_model.SemiParametericModel(para_model,nonpara_model,mode=mode,penalty=penalty)
print('==> Model already !')

# optimal
print('==> Optimal start...')
optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
train_process = RCD_compa_optimal.OptimalProcess(
    train=train,test=test,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
    tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device=device)
ret = train_process.run()
print('==> Optimal finish...')
train_loss = ret["train_loss_mat"]
test_loss = ret["test_loss_mat"]
parameter_value1 = ret["parameter_mat1"]
parameter_value2 = ret["parameter_mat2"]
parameter_value3 = ret["parameter_mat3"]
#print(train_loss)
#print(test_loss)
#print(parameter_value)
#print(parameter_value2)
save_mat = np.array([train_loss,test_loss,parameter_value1,parameter_value2,parameter_value3])
np.save(path_save,save_mat)



