# main function
import torch
import os
import numpy as np


import torch.nn as nn
from ignite.utils import convert_tensor
import shelve

import RD_dataset
import RD_model
import RD_optimal

import copy

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


for current_seed in range(args.train_seed,(args.train_seed + 25)):
    print('current seed %d'%(current_seed))
    # setting parameters
    D = 3e-3
    func_f = None
    func_g = None
    bc = None
    batch_size = [1,1]
    train_curve_num= 64
    test_curve_num= round(train_curve_num/4)
    utest_curve_num= round(train_curve_num/4)
    sample_size= args.t_size
    T_range=[0,2.5]
    delta_t=1e-3
    range_grid=[[-1,1]]
    period=True
    num_grid=args.sample_size
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
    nupdate = 64*5
    nepoch = 25
    nlog= 64
    path_train = "./data_para/rd_train_%d_%d_%d_%.3f_%.3f_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,num_grid,seed_train)
    path_test = "./data_para/rd_test_%d_%d_%d_%.3f_%.3f_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,num_grid,seed_train)
    path_utest = "./data_para/rd_utest_%d_%d_%d_%.3f_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,D,num_grid,seed_train)
    path_save = "./outputs_para/rd_variance_%d_%d_%d_%.3f_%.3f_%d_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,num_grid,seed_train)

    path_checkpoint = "./checkpoint_para/ckpt_rd_variance_%d_%d_%d_%.3f_%.3f_%d_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,num_grid,seed_train)
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
    
    
    train,test,utest = RD_dataset.create_data(path_train=path_train,path_test=path_test,path_utest=path_utest,
        D=D,func_f=func_f,bc=bc,
        batch_size=batch_size,train_curve_num=train_curve_num,test_curve_num=test_curve_num,
        sample_size=sample_size,T_range=T_range,delta_t=delta_t,range_grid=range_grid,
        period=period,num_grid=num_grid,seed_train=seed_train,seed_test=seed_test,seed_utest=seed_utest,
        initial=initial,noise_sigma=noise_sigma,dim=dim,rand=False)
    print('==> Data already !')

    # construct model
    print('==> Preparing model..')
    para_model = RD_model.ParametricPart(dx=(range_grid[0][1]-range_grid[0][0])/num_grid)
    para_model0 = RD_model.ParametricPart(dx=(range_grid[0][1]-range_grid[0][0])/num_grid)
    # nonpara_model = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
    nonpara_model0 = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
    checkpoints = torch.load("./checkpoint_para/ckpt_rd_%d_%d_%d_%.3f_%.3f_%d_%d.pth"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,D,num_grid,seed_train))
    # nonpara_model.net.load_state_dict(copy.deepcopy(checkpoints['net']))
    nonpara_model0.net.load_state_dict(copy.deepcopy(checkpoints['net']))
    para_model._params['D'] = nn.Parameter(torch.tensor(checkpoints['D'].to('cpu').detach().numpy()*1e-3))
    para_model0._params['D'] = nn.Parameter(torch.tensor(checkpoints['D'].to('cpu').detach().numpy()*1e-3+1e-5))
    mode = args.mode
    # if mode == 'None':
    #     mode = None
    # if mode is None:
    #     path_save = "./outputs/rd_incom_variance_%d_%d_%d_%.3f_%.3f_%d.npy"%(train_curve_num,test_curve_num,
    #                                                                 sample_size,noise_sigma,D,seed_train)

    #     path_checkpoint = "./checkpoint/ckpt_rd_incom_variance_%d_%d_%d_%.3f_%.3f_%d.pth"%(train_curve_num,test_curve_num,
    #                                                                 sample_size,noise_sigma,D,seed_train)
    penalty =None# 'orthogonal'
    #whole model
    full_model_dic = dict()
    nonpara_model = dict()
    for i in range(64):
        nonpara_model[str(i)] = RD_model.NonParametricModel(state_dim=dim,hidden=hidden)
        nonpara_model[str(i)].net.load_state_dict(copy.deepcopy(checkpoints['net']))
        full_model_dic[str(i)] = RD_model.SemiParametericModel(para_model,nonpara_model[str(i)],mode=mode,penalty=penalty)
    # full_model = RD_model.SemiParametericModel(para_model,nonpara_model,mode=mode,penalty=penalty)
    full_model0 = RD_model.SemiParametericModel(para_model0,nonpara_model0,mode=mode,penalty=penalty)
    print('==> Model already !')


    # optimal
    print('==> Optimal start...')
    optimizer = dict()
    for i in range(64):
        optimizer[str(i)] = torch.optim.Adam(nonpara_model[str(i)].parameters(), lr=tau_1*1e-4, betas=(0.9, 0.999))
    # optimal process
    
    # define optimal process 
    class OptimalProcess(object):
        def __init__(
            self, train, net, net0, optimizer, lambda_0, tau_2, niter=1, 
            nupdate=100, nlog=10, test=None, utest=None, nepoch=10, path='./checkpint/ckpt.pth', device="cpu"):

            self.device = device
            self.train = train
            self.test = test
            self.utest = utest
            self.nepoch = nepoch
        
            self.traj_loss = nn.MSELoss()
            self.net = dict()
            for i in range(64):
                self.net[str(i)] = net[str(i)].to(self.device)
            self.net0 = net0.to(self.device)
            self.optimizer = optimizer

            self.tau_2 = tau_2
            self.niter = niter
            self._lambda = lambda_0

            self.nlog = nlog
            self.nupdate = nupdate
            self.path = path

        def zero_grad(self):
            for optimizer in self.optimizers():
                optimizer.zero_grad()        

        def train_step(self, batch):
            #self.training()
            batch = convert_tensor(batch, self.device)
            loss, output = self.step(batch)
            with torch.no_grad():
                pass
            return batch, output, loss

        def val_step(self, batch, val=False):
            #self.evaluating()
            with torch.no_grad():
                batch = convert_tensor(batch, self.device)
                loss, output = self.step(batch, backward=False)

            return batch, output, loss

        def log(self, epoch, iteration, metrics):
            message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
                step=epoch *len(self.train)+ iteration+1,
                epoch=epoch+1,
                max_epoch=self.nepoch,
                i=iteration+1,
                max_i=len(self.train)
            )
            for name, value in metrics.items():
                message += ' | {name}: {value:.2e}'.format(name=name, value=value)
                
            print(message)

        def lambda_update(self, loss):
            self._lambda = self._lambda + self.tau_2 * loss
        
        def _forward(self, states, t, backward=True):
            # target = states
            y0 = states[:,0,:,:]
            with torch.no_grad():
                pred0 = self.net0(y0,t)
            pred = self.net[str(self.i)](y0, t)
            loss = self.traj_loss(pred, pred0)*1e10
            #penalty = self.net.get_penalty(states)

            if backward:
                loss_total = loss #* self._lambda #+ penalty
                loss_total.backward()
                self.optimizer[str(self.i)].step()
                self.optimizer[str(self.i)].zero_grad()

            loss = {
                'loss': loss,
            }

            output = {
                'states_pred'     : pred,
            }
            return loss, output

        def step(self, batch, backward=True):
            states = batch['states']
            t = batch['t'][0]
            loss, output = self._forward(states, t, backward)
            return loss, output

        def run(self):
            train_loss_mat = []
            test_loss_mat = []
            parameter_mat = []
            loss_test_last = torch.inf
            for epoch in range(self.nepoch): 
                for iteration, data in enumerate(self.train, 0):
                    self.i = iteration
                    for _ in range(self.niter):
                        _, _, loss = self.train_step(data)

                    total_iteration = epoch * ((len(self.train))) + (iteration + 1)
                    loss_train = loss['loss'].item()
                        # self.lambda_update(loss_train)

                    if total_iteration % self.nlog == 0:
                        self.log(epoch, iteration, loss)

                    if total_iteration % self.nupdate == 0:
                        with torch.no_grad(): 
                            loss_test = 0.
                            for j, data_test in enumerate(self.train, 0):
                                self.i = j
                                _, _, loss = self.val_step(data_test)
                                loss_test += loss['loss'].item()
                                
                            loss_test /= j + 1
                            loss_test = loss_test * (sample_size + 1) / sample_size

                            loss_test = {
                                'loss_test': loss_test,
                            }
                            print('#' * 80)
                            self.log(epoch, iteration, loss_test)
                            # print(f'lambda: {self._lambda}')
                            # print(f'D:{self.net.para_model.params["D"]}')
                            print('#' * 80)
                            # train_loss_mat.append(loss_train)
                            # test_loss_mat.append(loss_test["loss_test"])
                            # parameter_mat.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model.params["D"])).numpy().copy())
                            if loss_test["loss_test"] <= loss_test_last:
                                # u_loss_test = 0
                                # for j, data_test in enumerate(self.utest, 0):
                                #     _, _, u_loss = self.val_step(data_test)
                                #     u_loss_test += u_loss['loss'].item()
                                
                                # u_loss_test /= j + 1
                                # u_loss_test = {
                                #     'u_loss_test': u_loss_test,
                                # }
                                # print('#' * 80)
                                # self.log(epoch, iteration, u_loss_test)
                                # print('#' * 80)
                                print('saving...')
                                state = {
                                    'test_loss':loss_test["loss_test"],
                                    'epoch':epoch,
                                    'seed':torch.initial_seed(),
                                    # 'net':self.net.nonpara_model.net.state_dict(),
                                    # 'u_loss':u_loss_test["u_loss_test"],
                                }
                                torch.save(state, self.path)
                                loss_test_last = loss_test["loss_test"]
            return {"train_loss_mat":train_loss_mat,"test_loss_mat":test_loss_mat,"parameter_mat":parameter_mat}
        
        
        
    train_process = OptimalProcess(
        train=train,test=test,utest=utest,net=full_model_dic,net0=full_model0,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device = device)
    ret = train_process.run()
    print('==> Optimal finish...')
    # train_loss = ret["train_loss_mat"]
    # test_loss = ret["test_loss_mat"]
    # parameter_value = ret["parameter_mat"]
    # save_mat = np.array([train_loss,test_loss,parameter_value])
    # np.save(path_save,save_mat)


