# optimal process

import torch
import torch.nn as nn
import os, sys
import collections
from ignite.utils import convert_tensor
from nuisance_function import Logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# define optimal process 
class OptimalProcess(object):
    def __init__(
        self, train, net, optimizer, lambda_0, tau_2, niter=1, 
        nupdate=100, nlog=10, test=None, nepoch=10, path='./exp', device="cpu"):

        self.device = device
        self.path = path
        self.logger = Logger(filename=os.path.join(self.path, 'log.txt'))
        print(' '.join(sys.argv))

        self.train = train
        self.test = test
        self.nepoch = nepoch
        
        self.traj_loss = nn.MSELoss()
        self.net = net.to(self.device)
        self.optimizer = optimizer

        self.tau_2 = tau_2
        self.niter = niter
        self._lambda = lambda_0

        self.nlog = nlog
        self.nupdate = nupdate
    
    #def training(self, mode=True):
    #    for m in self.modules():
    #        m.train(mode)

    #def evaluating(self):
    #    self.training(mode=False)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    #def to(self, device):
    #    for m in self.modules():
    #        m.to(device)
     #   return self
    
    #def modules(self):
    #    for name, module in self.named_modules():
    #        yield module

    #def named_modules(self):
    #    for name, module in self._modules.items():
    #        yield name, module

    #def datasets(self):
    #    for name, dataset in self.named_datasets():
    #        yield dataset

    #def named_datasets(self):
    #    for name, dataset in self._datasets.items():
    #        yield name, dataset

    #def optimizers(self):
    #    for name, optimizer in self.named_optimizers():
    #        yield optimizer

    #def named_optimizers(self):
    #    for name, optimizer in self._optimizers.items():
    #        yield name, optimizer

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

    #def __setattr__(self, name, value):
    #    if isinstance(value, nn.Module):
    #        if not hasattr(self, '_modules'):
     #           self._modules = collections.OrderedDict()
     #       self._modules[name] = value
     #   elif isinstance(value, DataLoader):
    #        if not hasattr(self, '_datasets'):
     #           self._datasets = collections.OrderedDict()
     #       self._datasets[name] = value
     #   elif isinstance(value, Optimizer):
     #       if not hasattr(self, '_optimizers'):
     #           self._optimizers = collections.OrderedDict()
     #       self._optimizers[name] = value
     #   else:
     #       object.__setattr__(self, name, value)

    #def __getattr__(self, name):
    #    if '_modules' in self.__dict__:
    #        modules = self.__dict__['_modules']
    #        if name in modules:
    #            return modules[name]
    #    if '_datasets' in self.__dict__:
    #        datasets = self.__dict__['_datasets']
    #        if name in datasets:
    #            return datasets[name]
    #    if '_optimizers' in self.__dict__:
    #        optimizers = self.__dict__['_optimizers']
    #        if name in optimizers:
    #            return optimizers[name]
    #    raise AttributeError("'{}' object has no attribute '{}'".format(
    #        type(self).__name__, name))

    #def __delattr__(self, name):
    #    if name in self._modules:
    #        del self._modules[name]
    #    elif name in self._datasets:
    #        del self._datasets[name]
     #   elif name in self._optimizers:
    #        del self._optimizers[name]
     #   else:
     #       object.__delattr__(self, name)

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
        target = states
        y0 = states[:,0,:,:,:]
        pred = self.net(y0, t)
        loss = self.traj_loss(pred, target)
        #penalty = self.net.get_penalty(states)

        if backward:
            loss_total = loss #* self._lambda #+ penalty
            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss = {
            'loss': loss,
            #'penalty': penalty,
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
        parameter_mat2 = []
        for epoch in range(self.nepoch): 
            for iteration, data in enumerate(self.train, 0):
                for _ in range(self.niter):
                    _, _, loss = self.train_step(data)

                total_iteration = epoch * (len(self.train)) + (iteration + 1)
                loss_train = loss['loss'].item()
                self.lambda_update(loss_train)

                if total_iteration % self.nlog == 0:
                    self.log(epoch, iteration, loss)

                if total_iteration % self.nupdate == 0:
                    with torch.no_grad(): 
                        loss_test = 0.
                        for j, data_test in enumerate(self.test, 0):
                            _, _, loss = self.val_step(data_test)
                            loss_test += loss['loss'].item()
                            
                        loss_test /= j + 1

                        #if loss_test_min == None or loss_test_min > loss_test:
                        #    loss_test_min = loss_test
                        #    torch.save({
                        #        'epoch': epoch,
                        #        'model_state_dict': self.net.state_dict(),
                        #        'optimizer_state_dict': self.optimizer.state_dict(),
                        #        'loss': loss_test_min, 
                        #    }, self.path + f'/model_{loss_test_min:.3e}.pt')
                        loss_test = {
                            'loss_test': loss_test,
                        }
                        print('#' * 80)
                        self.log(epoch, iteration, loss_test)
                        #print(f'lambda: {self._lambda}')
                        print(f'D_1:{self.net.para_model.params["D_1"]}')
                        print(f'D_2:{self.net.para_model.params["D_2"]}')
                        print('#' * 80)
                        train_loss_mat.append(loss_train)
                        test_loss_mat.append(loss_test["loss_test"])
                        parameter_mat.append(torch.Tensor.detach(self.net.para_model.params["D_1"]).numpy().copy())
                        parameter_mat2.append(torch.Tensor.detach(self.net.para_model.params["D_2"]).numpy().copy())
        return {"train_loss_mat":train_loss_mat,"test_loss_mat":test_loss_mat,"parameter_mat":parameter_mat,"parameter_mat2":parameter_mat2}