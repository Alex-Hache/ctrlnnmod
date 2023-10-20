import os
import sys

import numpy as np
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt

from nssmid.preprocessing import *
from nssmid.ss_integrators import *
from nssmid.ss_models import *
from nssmid.train import *
from nssmid.matlab.matlabfcn import *
from nssmid.losses import *
from nssmid.postprocessing import *
from nssmid.layers import *
import geotorch as geo


if __name__=='__main__':


    dMat = loadmat('data/sdp_exemple/data.mat')
    c = dMat['c']
    M0 = dMat['M0']
    cA = dMat['cAi']
    vA = []
    for k in range(cA.shape[1]):
        vA.append(torch.Tensor(cA[0][k]).to(dtype = torch.float32))
    nx = c.shape[1]



    prob = SDP_problem(nx,c,vA, torch.Tensor(M0).to(dtype= torch.float32))
    # Setup optimizer
    params_net = list(prob.parameters())
    lr = 1e-3
    optimi = torch.optim.Adam([
        {'params': params_net,    'lr': lr} ], lr=lr)
    mu = 10
    vObj = []
    vDq = []
    vRatio = []
    print_freq = 100

    def optim_sdp(max_iter = 10000):
        n_u = 0
        no_decrease_counter = 0
        patience = 1000
        tol_change = 1e-3 # Taille de la boule
        best_obj = math.inf
        for i in range(max_iter):
            objective, dQ, M = prob()
            optimi.zero_grad()
            L = objective + mu*torch.max(dQ)
            L.backward() 
            optimi.step()

            obj = objective.detach().squeeze().numpy()
            r = torch.max(dQ).detach().numpy()
            vObj.append(obj)
            vDq.append(r)
            if r==0: # Q is DD+
                # Start counting
                if  obj <  best_obj- tol_change:
                    best_obj = obj
                    no_decrease_counter = 0
                    
                else:
                    no_decrease_counter = no_decrease_counter +1
                if no_decrease_counter> patience:
                    with torch.no_grad():
                        #print("Updating U")
                        prob.layer.updateU_(M)
                        n_u = n_u +1
                    no_decrease_counter = 0
            if i%print_freq ==0:
                print(f"Iter {i} : Loss  = {float(L.detach().numpy()):.5f} | Distance to DD+ : {r:.5f} | No decrease count {no_decrease_counter} ")
            
        return prob.x, n_u

    x_opt, nu = optim_sdp(max_iter=120000) 

    print(f"x Optimal value is : {x_opt}")
    print(f"Number of bases updates : {nu}")


    f, ((ax1), (ax2)) = plt.subplots(2, 1,sharex=True)
    ax1.plot(vDq)
    ax2.plot(vObj)

    plt.show()



