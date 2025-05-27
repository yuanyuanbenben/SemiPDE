# main function
import torch
import os
import numpy as np

import GFN_dataset
import GFN_model
import GFN_optimal

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--sigma','-sigma','-s',default = 0.1, type = float, help='variance of noise')
parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--train_seed','-seed',default = 0, type = int, help='train seed')
parser.add_argument('--mode','-mode','-m',default = None, type = str, help='mode')
parser.add_argument('--sample_size','-size','-ss',default = 64, type = int, help='sample size')
parser.add_argument('--t_size','-tsize','-ts',default = 25, type = int, help='t size')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids


for current_seed in range(args.train_seed,(args.train_seed + 50)):
    print('current seed %d'%(current_seed))
    # setting parameters
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
    test_curve_num= round(train_curve_num/4)
    utest_curve_num= round(train_curve_num/4)
    sample_size = args.t_size
    T_range=[0,2.5]
    delta_t=1e-3
    range_grid=[[-1,1],[-1,1]]
    period=[True,True]
    num_grid=[args.sample_size,args.sample_size]
    seed_train=current_seed
    seed_test=current_seed + 100
    seed_utest=current_seed + 200
    initial=[0,1]
    noise_sigma=args.sigma
    dim=2
    hidden=[16,64,64,16,dim]
    lambda_0 = 1.0
    tau_1 = 1e-3
    tau_2 = 1
    niter = 5
    nupdate = 20
    nepoch = 300
    nlog= 4
    path_train = "./data_para/gfn_train_%d_%d_%d_%.3f_%.3f_%.3f_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2,num_grid[0],seed_train)
    path_test = "./data_para/gfn_test_%d_%d_%d_%.3f_%.3f_%.3f_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2,num_grid[0],seed_train)
    path_utest = "./data_para/gfn_utest_%d_%d_%d_%.3f_%.3f_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,D_1,D_2,num_grid[0],seed_train)
    path_save = "./outputs_para/gfn_%d_%d_%d_%.3f_%.3f_%.3f_%d_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2,num_grid[0],seed_train)

    path_checkpoint = "./checkpoint_para/ckpt_gfn_%d_%d_%d_%.3f_%.3f_%.3f_%d_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D_1,D_2,num_grid[0],seed_train)
    tor_1 = 1e-4
    tor_2 = 1e-4
    iteration= 30
    pertu_loss= 1e-5
    pertu_grad= 1e-5
    stepsize= 1e-5
    epsilon= 1e-2
    do = True
    arti_score = True
    if not os.path.isdir('data_para'):
        os.mkdir('data_para')
    if not os.path.isdir('outputs_para'):
        os.mkdir('outputs_para')
    if not os.path.isdir('checkpoint_para'):
        os.mkdir('checkpoint_para')
        
    # device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # generate data
    print('==> Preparing data..')
    train, test, utest = GFN_dataset.create_data(path_train=path_train,path_test=path_test,path_utest=path_utest,
        D_1=D_1,D_2=D_2,func_f=func_f,func_g=func_g,a=a,epsilon=epsilon,gamma=gamma,bc=bc,
        batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
        sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
        period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,seed_utest=seed_utest,
        initial=initial,noise_sigma=noise_sigma,dim=dim)
    print('==> Data already !')

    # construct model
    print('==> Preparing model..')
    para_model = GFN_model.ParametricPart(dx=(range_grid[0][1]-range_grid[0][0])/num_grid[0])
    nonpara_model = GFN_model.NonParametricModel(state_dim=dim,hidden=hidden,device=device)
    mode = args.mode
    # if mode == 'None':
    #     mode = None
    # if mode is None:
    #     path_save = "./outputs/rd2_incom_%d_%d_%d_%.3f_%.3f_%d.npy"%(train_curve_num,test_curve_num,
    #                                                                 sample_size,noise_sigma,D,seed_train)

    #     path_checkpoint = "./checkpoint/ckpt_rd2_incom_%d_%d_%d_%.3f_%.3f_%d.pth"%(train_curve_num,test_curve_num,
    #                                                                 sample_size,noise_sigma,D,seed_train)
    penalty = None
    #whole model
    full_model = GFN_model.SemiParametericModel(para_model,nonpara_model,mode=mode,penalty=penalty,device=device)
    print('==> Model already !')


    # optimal
    print('==> Optimal start...')
    optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
    train_process = GFN_optimal.OptimalProcess(
        train=train,test=test,utest=utest,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,device=device,tor=tor_1,path=path_checkpoint)
    ret = train_process.run()
    print('==> Optimal finish...')
    train_loss = ret["train_loss_mat"]
    test_loss = ret["test_loss_mat"]
    parameter_value = ret["parameter_mat"]
    parameter_value2 = ret["parameter_mat2"]
    save_mat = np.array([train_loss,test_loss,parameter_value,parameter_value2])
    np.save(path_save,save_mat)


