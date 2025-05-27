# main function
import torch
import os
import numpy as np

import NS_dataset
import NS_model
import NS_optimal

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--sigma','-sigma','-s',default = 0.1, type = float, help='variance of noise')
parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--train_seed','-seed',default = 0, type = int, help='train seed')
parser.add_argument('--sample','-sample','-n',default = 64, type = int, help='sample size')
parser.add_argument('--mode','-mode','-m',default = None, type = str, help='sample size')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

for current_seed in range(args.train_seed,(args.train_seed + 10)):
    print('current seed %d'%(current_seed))
    # setting parameters
    # data
    D_1 = [6e-3,7e-3,8e-3]
    D_2 = [3e-3,4e-3,5e-3]
    func_f = None
    func_g = None
    bc = "periodic"
    batch_size = [8,2]
    train_curve_num= args.sample
    test_curve_num= round(train_curve_num/4)
    utest_curve_num= round(train_curve_num/4)    
    sample_size=25
    T_range=[0,2.5]
    delta_t=2e-2
    range_grid=[[-1,1],[-1,1],[-1,1]]
    period=[True,True,True]
    # num_grid=[16,16,16]
    num_grid=[8,8,8]
    seed_train=current_seed
    seed_test=current_seed + 100
    seed_utest=current_seed + 200
    initial=[0,1]
    noise_sigma=args.sigma
    dim=3

    # net
    state_dim = [3,6]
    hidden=[16,64,64,16]
    lambda_0 = 0.1#5*64*25
    tau_1 = 1e-3
    tau_2 = 0.01
    niter = 5
    nupdate = 500
    nepoch = 200
    nlog= 100
    dx = (range_grid[0][1]-range_grid[0][0])/num_grid[0]
    path_train = "./data/ns2_train_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
    path_test = "./data/ns2_test_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
    path_utest = "./data/ns2_utest_%d_%d_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,seed_train)
    path_save = "./outputs/ns2_%d_%d_%d_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)

    path_checkpoint = "./checkpoint/ckpt_ns2_%d_%d_%d_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        
    # device = torch.device("cuda:0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generate data
    print('==> Preparing data..')
    train,test,utest = NS_dataset.create_data(path_train=path_train,path_test=path_test,path_utest=path_utest,
        D_1=D_1,D_2=D_2,func_f=func_f,func_g = func_g,bc=bc,
        batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
        sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
        period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,seed_utest=seed_utest,
        initial=initial,noise_sigma=noise_sigma,dim=dim)
    print('==> Data already !')

    # construct model
    print('==> Preparing model..')
    para_model = NS_model.ParametricPart(dx=dx,device=device)
    nonpara_model = NS_model.NonParametricModel(dx=dx,state_dim=state_dim,hidden=hidden,device=device)
    mode = args.mode
    if mode == 'None':
        mode = None
    if mode is None:
        path_save = "./outputs/ns2_incom_%d_%d_%d_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)

        path_checkpoint = "./checkpoint/ckpt_ns2_incom_%d_%d_%d_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
    penalty = None# 'orthogonal'
    #whole model
    full_model = NS_model.SemiParametericModel(para_model,nonpara_model,mode=mode,penalty=penalty)
    print('==> Model already !')

    # optimal
    print('==> Optimal start...')
    optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
    train_process = NS_optimal.OptimalProcess(
        train=train,test=test,utest=utest,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device=device)
    ret = train_process.run()
    print('==> Optimal finish...')
    train_loss = ret["train_loss_mat"]
    test_loss = ret["test_loss_mat"]
    parameter_value1 = ret["parameter_mat1"]
    parameter_value2 = ret["parameter_mat2"]
    parameter_value3 = ret["parameter_mat3"]
    parameter_value4 = ret["parameter_mat4"]
    parameter_value5 = ret["parameter_mat5"]
    parameter_value6 = ret["parameter_mat6"]
    #print(train_loss)
    #print(test_loss)
    #print(parameter_value)
    #print(parameter_value2)
    save_mat = np.array([train_loss,test_loss,parameter_value1,parameter_value2,parameter_value3,parameter_value4,parameter_value5,parameter_value6])
    np.save(path_save,save_mat)
