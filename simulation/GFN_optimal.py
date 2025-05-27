# optimal process
import torch
import torch.nn as nn
from ignite.utils import convert_tensor


# define optimal process 
class OptimalProcess(object):
    def __init__(
        self, train, net, optimizer, lambda_0, tau_2, niter=1, 
        nupdate=100, nlog=10, test=None, utest=None, nepoch=10, device="cpu", tor = 1e-3,path='./checkpint/ckpt.pth'):

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
        self.tor = tor
        self.path = path
    
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
    
    def _forward(self, states, t, backward=True):
        target = states
        y0 = states[:,0,:,:,:]
        pred = self.net(y0, t)
        loss = self.traj_loss(pred, target)
        penalty = 0

        if backward:
            penalty = self.net.get_penalty(states)
            loss_total = loss + penalty
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
                        loss_test = 0
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
                        print(f'D_1:{self.net.para_model.params["D_1"]*1e3}e-3')
                        print(f'D_2:{self.net.para_model.params["D_2"]*1e3}e-3')
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
                                'net':self.net.nonpara_model.net.state_dict(),
                                'u_loss':u_loss_test["u_loss_test"],
                            }
                            torch.save(state, self.path)
                            loss_test_last = loss_test["loss_test"]

        return {"train_loss_mat":train_loss_mat,"test_loss_mat":test_loss_mat,"parameter_mat":parameter_mat,"parameter_mat2":parameter_mat2}


# class OrthogonalTackingProcess(OptimalProcess):
#     def __init__(self, train, train_2, net, optimizer, lambda_0, tau_2, niter=1, 
#         nupdate=100, nlog=10, test=None, nepoch=10, path='./exp', 
#         device="cpu",tor=1e-3,iteration=100,pertu_loss=1e-5,pertu_grad = 1e-5,stepsize=1e-3,epsilon=1e-3):
#         super(OrthogonalTackingProcess, self).__init__(train, net, optimizer, lambda_0, tau_2, niter, nupdate, nlog, test, nepoch, path, device,tor)
#         self.iteration = iteration
#         self.train_2 = train_2
#         self.pertu_1 = pertu_loss
#         self.pertu_2 = pertu_grad 
#         self.stepsize = stepsize
#         self.epsilon = epsilon
#         # para
#         #self.D_1 = 0.
#         #self.D_2 = 0.
#         #self.D_1_pertu = 0.
#         #self.D_2_pertu = 0.
#         #self.D_1_current = 0.
#         #self.D_2_current = 0.

#     def _get_value(self, states, t):
#         y0 = states[:,0,:,:,:]
#         return self.net(y0, t)

#     def run(self):
#         loss_test_last = -self.tor - 1
#         for epoch in range(self.nepoch): 
#             for iteration, data in enumerate(self.train, 0):
#                 for _ in range(self.niter):
#                     _, _, loss = self.train_step(data)

#                 total_iteration = epoch * (len(self.train)) + (iteration + 1)
#                 loss_train = loss['loss'].item()
#                 self.lambda_update(loss_train)

#                 #if total_iteration % self.nlog == 0:
#                 #    self.log(epoch, iteration, loss)

#                 if total_iteration % self.nupdate == 0:
                    
#                     with torch.no_grad(): 
#                         loss_test = 0
#                         for j, data_test in enumerate(self.test, 0):
#                             _, _, loss = self.val_step(data_test)
#                             loss_test += loss['loss'].item()
                            
#                         loss_test /= j + 1
#                         loss_test = {
#                             'loss_test': loss_test,
#                         }
#                         print('#' * 80)
#                         self.log(epoch, iteration, loss_test)
#                         print('#' * 80)
#                         if abs(loss_test["loss_test"]/loss_test_last - 1) < self.tor:
#                             return 0
#                         loss_test_last = loss_test["loss_test"]

#         return 0

#     def _score_theta_batch(self, states, t, u_theta_f, state_dict_1,state_dict_2,require_variance=False):
#         # u(z;theta,f) --> u_theta_f
#         # para part
#         self.net.para_model._params["D_1"] = self.D_1_pertu
#         u_theta_pertu1_f = self._get_value(states, t)
#         self.net.para_model._params["D_1"] = self.D_1_current
#         u_d_theta_1_f = (u_theta_pertu1_f - u_theta_f) / self.pertu_1

#         self.net.para_model._params["D_2"] = self.D_2_pertu
#         u_theta_pertu2_f = self._get_value(states, t)
#         self.net.para_model._params["D_2"] = self.D_2_current
#         u_d_theta_2_f = (u_theta_pertu2_f - u_theta_f) / self.pertu_1

#         # nonpara part
#         self.net.nonpara_model.net.load_state_dict(state_dict_1)
#         u_theta_f_pertu1 = self._get_value(states, t)
#         u_theta_d_f_1 = (u_theta_f_pertu1 - u_theta_f) / self.pertu_1
#         self.net.nonpara_model.net.load_state_dict(state_dict_2)
#         u_theta_f_pertu2 = self._get_value(states, t)
#         u_theta_d_f_2 = (u_theta_f_pertu2 - u_theta_f) / self.pertu_1
#         self.net.nonpara_model.net.load_state_dict(self.model_state_dict)

#         error = states - u_theta_f
#         if require_variance:
#             return torch.tensor([
#                                  [
#                                     torch.mean(torch.square(error * (u_d_theta_1_f +  u_theta_d_f_1))),
#                                     torch.mean((error * (u_d_theta_1_f +  u_theta_d_f_1)) * (error * (u_d_theta_2_f+ u_theta_d_f_2)))
#                                  ],
#                                  [
#                                     torch.mean((error * (u_d_theta_1_f +  u_theta_d_f_1)) * (error * (u_d_theta_2_f+ u_theta_d_f_2))),
#                                     torch.mean(torch.square(error * (u_d_theta_2_f+ u_theta_d_f_2)))
#                                  ]
#                                  ]).float()

#         return {'score_1':torch.mean(error * (u_d_theta_1_f +  u_theta_d_f_1)),'score_2':torch.mean(error * (u_d_theta_2_f+ u_theta_d_f_2))}

#     def _score_theta(self, require_variance=False):
#         score_1 = 0.
#         score_2 = 0.
#         self.net.para_model._params["D_1"] = self.D_1_pertu
#         self.run()
#         state_dict_1 = self.net.nonpara_model.net.state_dict()
#         self.net.para_model._params["D_1"] = self.D_1_current
#         self.net.para_model._params["D_2"] = self.D_2_pertu
#         self.run()
#         state_dict_2 = self.net.nonpara_model.net.state_dict()
#         self.net.para_model._params["D_2"] = self.D_2_current
#         self.net.nonpara_model.net.load_state_dict(self.model_state_dict)
#         if require_variance:
#             score_square = torch.tensor([[0.,0.],[0.,0.]]).float()
#             for j, batch in enumerate(self.train_2,0):
#                 batch = convert_tensor(batch, self.device)
#                 states = batch['states']
#                 t = batch['t'][0]
#                 u_theta_f = self._get_value(states, t)
#                 # para part
#                 score_square = score_square + self._score_theta_batch(states, t, u_theta_f,state_dict_1,state_dict_2,True)
#             return score_square / (j + 1)

#         for j, batch in enumerate(self.train_2,0):
#             batch = convert_tensor(batch, self.device)
#             states = batch['states']
#             t = batch['t'][0]
#             u_theta_f = self._get_value(states, t)
#             # para part
#             ret = self._score_theta_batch(states, t, u_theta_f,state_dict_1,state_dict_2)
#             score_1 += ret['score_1']
#             score_2 += ret['score_2']
#         return torch.tensor([score_1,score_2]).float() / (j + 1)
 
#     def information_theta(self, require_variance=False):
#         self.D_1 = self.net.para_model._params["D_1"]
#         self.D_2 = self.net.para_model._params["D_2"]
#         # psi(z,theta,f)
#         self.D_1_current = self.D_1
#         self.D_2_current = self.D_2
#         self.D_1_pertu = self.D_1_current + self.pertu_1
#         self.D_2_pertu = self.D_2_current + self.pertu_1
#         self.model_state_dict = self.net.nonpara_model.net.state_dict()
#         if require_variance:
#             return self._score_theta(require_variance)
#         score_current = self._score_theta()
#         # psi(z,dtheta,df)
#         self.D_1_current = self.D_1 + self.pertu_2
#         self.D_1_pertu = self.D_1_current + self.pertu_1
#         self.run()
#         self.model_state_dict = self.net.nonpara_model.net.state_dict()
#         score_forward_1 = self._score_theta()
#         self.D_1_current = self.D_1
#         self.D_1_pertu = self.D_1_current + self.pertu_1
#         self.D_2_current = self.D_2 + self.pertu_2
#         self.D_2_pertu = self.D_2_current + self.pertu_1
#         self.run()
#         self.model_state_dict = self.net.nonpara_model.net.state_dict()
#         score_forward_2 = self._score_theta()
#         self.net.para_model._params["D_2"] = self.D_2
#         mat = torch.cat(((score_forward_1-score_current).view(1,2),(score_forward_2-score_current).view(1,2)),axis=0)/self.pertu_2
#         c = (mat[1,0] + mat[0,1])/2
#         mat[1,0] = c
#         mat[0,1] = c
#         mat[0,0] += 1e-7
#         mat[1,1] += 1e-7
#         return {'loss':score_current,'information_invert':torch.linalg.inv(mat)}
        
#     def variance(self, score_square, information_invert):
#         return torch.Tensor.detach(torch.mm(torch.mm(information_invert, score_square),information_invert)).numpy().copy()

#     def track_theta(self,only_variance=False):
#         if only_variance:
#             ret = self.information_theta()
#             information_invert = ret['information_invert']
#             score_square = self.information_theta(True)
#             sigma_square = self.variance(score_square,information_invert)
#             return {'variance':sigma_square}
#         score_last = torch.tensor([-1,-1]).float()
#         para_1_mat = []
#         para_2_mat = []
#         for iter in range(self.iteration):
#             self.pertu_1 = np.random.uniform(1e-6,1e-5)
#             self.pertu_2 = np.random.uniform(1e-6,1e-5)
#             ret = self.information_theta()
#             score = ret['loss']
#             information_invert = ret['information_invert']
#             grad_used_1 = torch.sum(information_invert[0,] * score)
#             grad_used_2 = torch.sum(information_invert[1,] * score)
#             if torch.sum(torch.abs(score/(score_last+1e-7) - 1)) < self.epsilon:
#                 score_square = self.information_theta(True)
#                 sigma_square = self.variance(score_square,information_invert)
#                 break
#             self.net.para_model._params["D_1"] = self.D_1 - grad_used_1
#             self.net.para_model._params["D_2"] = self.D_2 - grad_used_2
#             para_1_mat.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model._params["D_1"])).numpy().copy())
#             para_2_mat.append(torch.Tensor.detach(torch.Tensor.cpu(self.net.para_model._params["D_2"])).numpy().copy())
#             print('#' * 80)
#             print(f'Iterations:{iter}')
#             print(f'D_1:{self.net.para_model._params["D_1"]*1e3}e-3')
#             print(f'D_2:{self.net.para_model._params["D_2"]*1e3}e-3')
#             print(f'score:{score*1e4}e-4')
#             score_last = score
#             self.run()
#         score_square = self.information_theta(True)
#         sigma_square = self.variance(score_square,information_invert)
#         return {"para_1":para_1_mat, "para_2":para_2_mat,"variance":sigma_square}



