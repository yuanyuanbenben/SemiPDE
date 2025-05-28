# optimal process
import torch
import torch.nn as nn
from ignite.utils import convert_tensor

# define optimal process 
class OptimalProcess(object):
    def __init__(
        self, train, net, optimizer,  position, lambda_0, tau_2, niter=1, 
        nupdate=100, nlog=10, test=None, nepoch=10, path='./checkpint/ckpt.pth', device="cpu"):

        self.device = device
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
        self.path = path
        
        self.position = position
        
        

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    def train_step(self, batch):
        #self.training()
        batch = convert_tensor(batch, self.device)
        loss, output = self.step(batch)
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
    
    def _forward(self, states, t, init, x_position, backward=True):
        target = states
        self.net.nonpara_model.change_x_position(x_position)
        pred = self.net(init, t)
        loss = self.traj_loss(pred[:,:,self.position], target)# + self.traj_loss(pred[:,:,0],pred[:,:,-1])*100
        # penalty = self.net.get_penalty(states)

        if backward:
            #input_ = convert_tensor(torch.tensor([[x,y] for x in t for y in x_position[0,:]]).float(),self.device)
            loss_total = loss + self.traj_loss(pred[:,0,:],pred[:,-1,:])*10 #+ self.net.nonpara_model._loss(input_)*0.01
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
        init = batch['initial']
        x_position = batch['x_position']
        loss, output = self._forward(states, t, init, x_position, backward)
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
                        print(f'C_D:{self.net.para_model.params["C_D"],}')
                        print(f'nv:{self.net.para_model.params["nv"],}')
                        print('#' * 80)
                        train_loss_mat.append(loss_train)
                        test_loss_mat.append(loss_test["loss_test"])
                        parameter_mat.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model.params["C_D"])).numpy().copy())
                        parameter_mat2.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model.params["nv"])).numpy().copy())
                        if epoch >= 30:
                            if loss_test["loss_test"] <= loss_test_last:
                                state = {
                                    'C_D': self.net.para_model.params["C_D"],
                                    'nv': self.net.para_model.params["nv"],
                                    'test_loss':loss_test["loss_test"],
                                    'epoch':epoch,
                                    'seed':torch.initial_seed(),
                                    'net':self.net.nonpara_model.net.state_dict(),
                                    # 'net2':self.net.nonpara_model.net2.state_dict()
                                }
                                torch.save(state, self.path)
                                loss_test_last = loss_test["loss_test"]
        return {"train_loss_mat":train_loss_mat,"test_loss_mat":test_loss_mat,"parameter_mat":parameter_mat,"parameter_mat2":parameter_mat2}