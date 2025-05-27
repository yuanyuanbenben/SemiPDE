# main function
import torch
import os
import numpy as np

import RD_dataset
import RD_model
import RD_optimal

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--sigma','-sigma','-s',default = 0.1, type = float, help='variance of noise')
parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--train_seed','-seed',default = 0, type = int, help='train seed')
parser.add_argument('--sample','-sample','-n',default = 64, type = int, help='sample size')
parser.add_argument('--mode','-mode','-m',default = None, type = str, help='penalty mode')
parser.add_argument('--model_mode','-model','-mod',default = None, type = str, help='model type')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids


for current_seed in range(args.train_seed,(args.train_seed + 10)):
    print('current seed %d'%(current_seed))
    # setting parameters
    D = 3e-3
    func_f = None
    if args.model_mode == 'no_func':
        print('para model...')
        func_f = lambda u: 0
    bc = None
    batch_size = [16,4]
    train_curve_num= args.sample
    test_curve_num= round(train_curve_num/4)
    utest_curve_num= round(train_curve_num/4)
    sample_size=25
    T_range=[0,2.5]
    delta_t=1e-3
    range_grid=[[-1,1]]
    period=True
    num_grid=64
    # num_grid=32
    seed_train=current_seed
    seed_test=current_seed + 100
    seed_utest=current_seed + 200
    initial=[0,1]
    noise_sigma=args.sigma
    dim=1
    hidden=[16,64,64,16,dim]
    lambda_0 = 1.0
    tau_1 = 1e-3
    tau_2 = 1
    niter = 5
    nupdate = 20
    nepoch = 100
    nlog= 4
    path_train = "./data/rd2_train_%d_%d_%d_%.3f_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)
    path_test = "./data/rd2_test_%d_%d_%d_%.3f_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)
    path_utest = "./data/rd2_utest_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,D,seed_train)
    path_save = "./outputs/rd2_%d_%d_%d_%.3f_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)

    path_checkpoint = "./checkpoint/ckpt_rd2_%d_%d_%d_%.3f_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        
    if args.model_mode == 'no_func':
        if not os.path.isdir('data_paramodel'):
            os.mkdir('data_paramodel')
        if not os.path.isdir('outputs_paramodel'):
            os.mkdir('outputs_paramodel')
        if not os.path.isdir('checkpoint_paramodel'):
            os.mkdir('checkpoint_paramodel')
        path_train = "./data_paramodel/rd_train_%d_%d_%d_%.3f_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)
        path_test = "./data_paramodel/rd_test_%d_%d_%d_%.3f_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                        sample_size,noise_sigma,D,seed_train)
        path_utest = "./data_paramodel/rd_utest_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                        sample_size,D,seed_train)
        path_save = "./outputs_paramodel/rd_%d_%d_%d_%.3f_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                        sample_size,noise_sigma,D,seed_train)

        path_checkpoint = "./checkpoint_paramodel/ckpt_rd_%d_%d_%d_%.3f_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                        sample_size,noise_sigma,D,seed_train)
    # device = torch.device("cuda:0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generate data
    print('==> Preparing data..')
    train,test,utest = RD_dataset.create_data(path_train=path_train,path_test=path_test,path_utest=path_utest,
        D=D,func_f=func_f,bc=bc,
        batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
        sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
        period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,seed_utest=seed_utest,
        initial=initial,noise_sigma=noise_sigma,dim=dim)
    print('==> Data already !')

    # construct model
    print('==> Preparing model..')
    para_model = RD_model.ParametricPart(dx=(range_grid[0][1]-range_grid[0][0])/num_grid)
    nonpara_model = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
    mode = args.mode
    if mode == 'None':
        mode = None
    if mode is None:
        path_save = "./outputs/rd2_incom_%d_%d_%d_%.3f_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)

        path_checkpoint = "./checkpoint/ckpt_rd2_incom_%d_%d_%d_%.3f_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)
        if args.model_mode == 'no_func':
            path_save = "./outputs_paramodel/rd2_incom_%d_%d_%d_%.3f_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)

            path_checkpoint = "./checkpoint_paramodel/ckpt_rd2_incom_%d_%d_%d_%.3f_%.3f_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,seed_train)
    penalty =None# 'orthogonal'
    #whole model
    full_model = RD_model.SemiParametericModel(para_model,nonpara_model,mode=mode,penalty=penalty)
    print('==> Model already !')


    # optimal
    print('==> Optimal start...')
    optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
    train_process = RD_optimal.OptimalProcess(
        train=train,test=test,utest=utest,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device = device)
    ret = train_process.run()
    print('==> Optimal finish...')
    train_loss = ret["train_loss_mat"]
    test_loss = ret["test_loss_mat"]
    parameter_value = ret["parameter_mat"]
    save_mat = np.array([train_loss,test_loss,parameter_value])
    np.save(path_save,save_mat)


