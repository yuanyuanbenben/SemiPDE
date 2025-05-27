# main function
import torch
import os
import numpy as np

import torch.nn as nn
from ignite.utils import convert_tensor
import shelve

import NS_dataset
import NS_model
import NS_optimal

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--sigma','-sigma','-s',default = 0.1, type = float, help='variance of noise')
parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--train_seed','-seed',default = 0, type = int, help='train seed')
parser.add_argument('--sample','-sample','-n',default = 64, type = int, help='sample size')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

for current_seed in range(args.train_seed,(args.train_seed + 50)):
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
    hidden=[16,64,64,16,3]
    lambda_0 = 0.1#5*64*25
    tau_1 = 1e-3
    tau_2 = 0.01
    niter = 5
    nupdate = 80
    nepoch = 2000
    nlog= 16
    dx = (range_grid[0][1]-range_grid[0][0])/num_grid[0]
    path_train = "./data/ns2_train_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
    path_test = "./data/ns2_test_%d_%d_%d_%.3f_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)
    path_utest = "./data/ns2_utest_%d_%d_%d_%d"%(train_curve_num,test_curve_num,
                                                                    sample_size,seed_train)
    path_save = "./outputs/ns2_nonpara_%d_%d_%d_%.3f_%d.npy"%(train_curve_num,test_curve_num,
                                                                    sample_size,noise_sigma,seed_train)

    path_checkpoint = "./checkpoint/ckpt_ns2_nonpara_%d_%d_%d_%.3f_%d.pth"%(train_curve_num,test_curve_num,
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
    # test_data = torch.from_numpy(shelve.open(path_utest)[str(0)]).float().permute(0,2,3,4,1).contiguous().view(26*16*16*16,3)
    test_data = torch.from_numpy(shelve.open(path_utest)[str(0)]).float().permute(0,2,3,4,1).contiguous().view(26*8*8*8,3)
    # noise = torch.tensor(np.random.normal(scale=noise_sigma,size=(26*16*16*16,3)),dtype=torch.float32)
    noise = torch.tensor(np.random.normal(scale=noise_sigma,size=(26*8*8*8,3)),dtype=torch.float32)
    print('==> Data already !')
    full_model = NS_model.NonparaModel(state_dim=4,hidden=hidden)

    # optimal
    print('==> Optimal start...')
    optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
    t_index = np.linspace(0,2.5,26,endpoint=True)
    # x_index = np.linspace(-1,1,16,endpoint=False)
    x_index = np.linspace(-1,1,8,endpoint=False)
    state_index = np.array([[a,b,c,d] for a in t_index for b in x_index for c in x_index for d in x_index])
    state_index = torch.tensor(state_index,dtype=torch.float32).to(device)
    # define optimal process 
    class OptimalProcess(object):
        def __init__(
            self, test, true_test, net, optimizer, lambda_0, tau_2, niter=1, 
            nupdate=100, nlog=10,nepoch=10, path='./checkpint/ckpt.pth', device="cpu"):

            self.device = device
            self.test = test
            self.utest = true_test
            self.nepoch = nepoch
        
            self.traj_loss = nn.MSELoss()
            self.net = net.to(self.device)
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
                step=epoch + iteration+1,
                epoch=epoch+1,
                max_epoch=self.nepoch,
                i=iteration+1,
                max_i=1
            )
            for name, value in metrics.items():
                message += ' | {name}: {value:.2e}'.format(name=name, value=value)
                
            print(message)

        def lambda_update(self, loss):
            self._lambda = self._lambda + self.tau_2 * loss
        
        def _forward(self, states, backward=True):
            target = states
            y = self.net(state_index)
            loss = self.traj_loss(y, target)
            #penalty = self.net.get_penalty(states)

            if backward:
                loss_total = loss #* self._lambda #+ penalty
                loss_total.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss = {
                'loss': loss,
            }

            output = {
                'states_pred'     : y,
            }
            return loss, output

        def step(self, batch, backward=True):
            states = batch
            loss, output = self._forward(states, backward)
            return loss, output

        def run(self):

            for epoch in range(self.nepoch): 
                data = self.test
                for iteration in range(1):
                    for _ in range(self.niter):
                        _, _, loss = self.train_step(data)

                    total_iteration = epoch  + (iteration + 1)
                    loss_train = loss['loss'].item()
                    self.lambda_update(loss_train)
                    
                    if total_iteration % self.nlog == 0:
                        self.log(epoch, iteration, loss)

                    if total_iteration % self.nupdate == 0:
                        with torch.no_grad(): 
                            u_loss_test = 0
                            data_test = self.utest
                            for j in range(1):
                                _, _, u_loss = self.val_step(data_test)
                                u_loss_test += u_loss['loss'].item()
                            
                            u_loss_test /= j + 1
                            u_loss_test = {
                                'u_loss_test': u_loss_test,
                            }
                            print('#' * 80)
                            self.log(epoch, iteration, u_loss_test)
                            print('#' * 80)
                            state = {
                                'epoch':epoch,
                                'seed':torch.initial_seed(),
                                'net':self.net.net.state_dict(),
                                'u_loss':u_loss_test["u_loss_test"],
                            }
                            torch.save(state, self.path)
    train_process = OptimalProcess(
        test=test_data+noise,true_test=test_data,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
        tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path_checkpoint,device = device)
    ret = train_process.run()
    print('==> Optimal finish...')