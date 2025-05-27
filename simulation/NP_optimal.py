# optimal process
import torch
import torch.nn as nn
from ignite.utils import convert_tensor

# define optimal process 
class OptimalProcess(object):
    def __init__(
        self, train, net, optimizer, lambda_0, tau_2, niter=1, 
        nupdate=100, nlog=10, test=None, utest=None,nepoch=10, path='./checkpint/ckpt.pth', device="cpu"):

        self.device = device
        self.train = train
        self.test = test
        self.utest = utest
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
        penalty = self.net.get_penalty(states)

        if backward:
            loss_total = loss * self._lambda + penalty
            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

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
        parameter_mat2 = []
        loss_test_last = torch.inf
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
                        loss_test = {
                            'loss_test': loss_test,
                        }
                        print('#' * 80)
                        self.log(epoch, iteration, loss_test)
                        print(f'lambda: {self._lambda}')
                        print(f'D_1:{self.net.para_model.params["D_1"]*1e3}1e-3')
                        print(f'D_2:{self.net.para_model.params["D_2"]*1e3}1e-3')
                        print('#' * 80)
                        train_loss_mat.append(loss_train)
                        test_loss_mat.append(loss_test["loss_test"])
                        parameter_mat.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model.params["D_1"])).numpy().copy())
                        parameter_mat2.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model.params["D_2"])).numpy().copy())
                        if loss_test["loss_test"] <= loss_test_last:
                            u_loss_test = 0
                            for j, data_test in enumerate(self.utest, 0):
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
                                'D_1': self.net.para_model.params["D_1"],
                                'D_2': self.net.para_model.params["D_2"],
                                'test_loss':loss_test["loss_test"],
                                'epoch':epoch,
                                'seed':torch.initial_seed(),
                                'net1':self.net.nonpara_model.net1.state_dict(),
                                'net2':self.net.nonpara_model.net2.state_dict(),
                                'u_loss':u_loss_test["u_loss_test"],
                            }
                            torch.save(state, self.path)
                            loss_test_last = loss_test["loss_test"]
        return {"train_loss_mat":train_loss_mat,"test_loss_mat":test_loss_mat,"parameter_mat":parameter_mat,"parameter_mat2":parameter_mat2}