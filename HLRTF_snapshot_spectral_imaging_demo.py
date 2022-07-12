import torch
from torch import nn, optim 
from torch.autograd import Variable 
import os 
from utils import * 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt 
import scipy.io
import math
import time

expand = 1
shrink = 5
gamma = 0.00001
data_all = ["data/Bird"]
c_all = ["SR3"]

max_iter = 4000
lr_real = 0.003

class Y_net(nn.Module): 
    def __init__(self,n_1,n_2,n_3):
        super(Y_net, self).__init__()
        self.A_hat = nn.Parameter(torch.Tensor(n_3*expand,n_1,n_2//shrink))
        self.B_hat = nn.Parameter(torch.Tensor(n_3*expand,n_2//shrink,n_2))
        
        self.net = nn.Sequential(permute_change(1,2,0),
                                 nn.Linear(int(n_3*expand),int(n_3*expand),bias = False),
                                 nn.LeakyReLU(),
                                 nn.Linear(int(n_3*expand),n_3,bias = False))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A_hat.size(0))
        self.A_hat.data.uniform_(-stdv, stdv)
        self.B_hat.data.uniform_(-stdv, stdv)
                                    
    def forward(self):
        x = torch.matmul(self.A_hat,self.B_hat)
        return self.net(x)

for data in data_all:
    for c in c_all:
        F_norm = nn.MSELoss()
        
        file_name = data+c+'.mat'
        mat = scipy.io.loadmat(file_name)
        X_np = mat["Nhsi"]
        X = torch.from_numpy(X_np).type(dtype).cuda()
        
        mat = scipy.io.loadmat(file_name)
        mask_np = (mat["mask"])
        mask = torch.from_numpy(mask_np).type(dtype).cuda()
        model = Y_net(X.shape[0],X.shape[1],mask.shape[2]).type(dtype)
        
        params = []
        params += [x for x in model.parameters()]
        
        s = sum([np.prod(list(p.size())) for p in params]); 
        print('Number of params: %d' % s)
        optimizier = optim.Adam(params, lr=lr_real, weight_decay=10e-8) 
        
        t0 = time.time()
        show = [0,5,9]
        for iter in range(max_iter):
            X_Out_real = model()
            loss = F_norm(torch.sum(X_Out_real*mask,dim = 2),X)
            i = 0
            for p in params:
                i += 1
                if i == 1: # A  
                    loss += gamma*torch.norm(p[:,1:,:]-p[:,:-1,:],1)
                    
                if i == 4: # H_k
                    loss += gamma*torch.norm(p[1:,:]-p[:-1,:],1)
                    
                if i == 2: # B
                    loss += gamma*torch.norm(p[:,:,1:]-p[:,:,:-1],1)
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
        
            if iter % 100 == 0:
                print('iteration:',iter)
                plt.figure(figsize=(11,33))
                plt.subplot(121)
                plt.imshow(np.clip(np.stack((X.cpu().detach().numpy(),
                                     X.cpu().detach().numpy(),
                                     X.cpu().detach().numpy()),2),0,1))
                plt.title('Observed')
        
                plt.subplot(122)
                plt.imshow(np.clip(np.stack((X_Out_real[:,:,show[0]].cpu().detach().numpy(),
                                     X_Out_real[:,:,show[1]].cpu().detach().numpy(),
                                     X_Out_real[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('Recovered')
                plt.show()