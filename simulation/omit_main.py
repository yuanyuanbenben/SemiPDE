# main function
import torch
import GFN_dataset
import omit_GFN_model
import omit_GFN_optimal
import os
import numpy as np

# setting parameters
D_1 = 1e-3
D_2 = 5e-3
func_f = lambda u,w: u-u**3-5e-3-w
func_g = None
a = 0.1
epsilon = 1
gamma = 1
bc = None
batch_size = 64
train_curve_num=1600
test_curve_num=300
sample_size=25
T_range=[0,2.5]
delta_t=1e-3
range_grid=[[-1,1],[-1,1]]
period=[True,True]
num_grid=[64,64]
seed_train=None
seed_test=20221106
initial=[0,1]
noise_sigma=0.01
dim=2
hidden=[16,64,64,16]
lambda_0 = 1.0
tau_1 = 1e-3
tau_2 = 1
niter = 5
nupdate = 50
nepoch = 50000
nlog= 5
path = "./exp"
path_train = '_train'
path_test = '_test'
os.makedirs(path,exist_ok=True)

os.makedirs("./outputs",exist_ok=True)

device = torch.device("cuda:0")

# generate data
train,test = GFN_dataset.create_data(path_train=path_train,path_test=path_test,
    D_1=D_1,D_2=D_2,func_f=func_f,func_g=func_g,a=a,epsilon=epsilon,gamma=gamma,bc=bc,
    batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
    sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
    period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,
    initial=initial,noise_sigma=noise_sigma,dim=dim)

# construct model
para_model = omit_GFN_model.ParametricPart(dx=(range_grid[0][1]-range_grid[0][0])/num_grid[0])
nonpara_model = omit_GFN_model.NonParametricModel(state_dim=dim,hidden=hidden)
#whole model
full_model = omit_GFN_model.SemiParametericModel(para_model,nonpara_model)

# optimal
optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
train_process = omit_GFN_optimal.OptimalProcess(
    train=train,test=test,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
    tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path,device=device)
ret = train_process.run()
train_loss = ret["train_loss_mat"]
test_loss = ret["test_loss_mat"]
parameter_value = ret["parameter_mat"]
parameter_value2 = ret["parameter_mat2"]
print(train_loss)
print(test_loss)
print(parameter_value)
print(parameter_value2)
save_mat = np.array([train_loss,test_loss,parameter_value,parameter_value2])
np.save("./outputs/gfn_no_2.npy",save_mat)