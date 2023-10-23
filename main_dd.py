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
from nssmid.linalg_utils import *
from nssmid.lmis import *

if __name__=='__main__':
        # Set seed for reproductibility
        np.random.seed(27)
        torch.manual_seed(27)

        # Load data
        dMatTrain = loadmat('data_train_dd.mat')
        dMatTest = loadmat('data_train_dd.mat')

        # Model order 
        nx = 2
        # Preprocess data
        u_train, y_train, x_train, y_dot_train, ts = preprocess_mat_file(dMatTrain, nx)
        u_test, y_test, x_test, y_dot_test, ts = preprocess_mat_file(dMatTest, nx)
        nd = 1 # Number of disturbance inputs

        # Perform linear estimate from train data
        save_bla_path = os.path.join(os.getcwd(), 'BLA.mat')
        A0, B0, C0, D0 = findBLA(u_train, y_train, nx, ts= ts, model_type = 'continuous',
                                save= True, strNameSave = save_bla_path)

        # Setting networks dimensions
        nu = u_train.shape[1]
        ny = y_train.shape[1]

        lin_model = NNLinear(nu,nx, ny)
        lin_model.init_model_(A0, B0, C0)
        lin_sim = RK4Simulator(lin_model, ts)
        x_bla, y_bla = lin_sim.simulate(u_train, torch.zeros(nx)) 


        # Create datasets and dataloaders
        seq_len = 30
        batch_size = 512

        nh = 16 # Number of hidden neurons
        n_layers = 1

        # Creating models
        model = GRNSSM(nu,nh, nx, ny, n_hid_layers = 1)

        model.init_weights(A0, B0, C0, isLinTrainable = True) 
        # Define loss function

        # First define alpha stability criterion
        alpha = torch.max(torch.real(torch.linalg.eigvals(A0))).numpy()
        lmi = LMI_decay_rate(alpha,model.linmod.A.weight, epsilon = 1e-6)

        model_sim = RK4Simulator(model, ts)


        #criterion = Mixed_MSELOSS(alpha = 0)
        # mu = 1e-5
        criterion = Mixed_MSELOSS_RelaxedLMI(rel_lmi, eta= 0, kappa = 0)

        # Optimiser parameters
        strOptimizer = 'adam'
        lr = 0.1
        num_epochs = 25000


        # validation parameters
        patience = 1000
        tol_change = 1e-7
        test_freq = 100

        # Train
        best_model, dictRes = train_network(model_sim, u_train, y_train, nx , 
                u_test, y_test, batch_size, seq_len,
                lr, num_epochs, criterion, test_freq = 100, patience = patience)


        ''' 
        Check decay rate constraints gamma :
        '''
        lmi = LMI_DecayRate(best_model.ss_model.linmod.A.weight, alpha, epsilon = 1e-6)
        Z = torch.eye((2*nx))
        rel_lmi = RelaxedLMI(lmi, Z)
        bP_SDP = isSDP(lmi.P.weight)
        bLyap_SDP = isSDP(lmi())
        bDecayRate = torch.all(getEigenvalues(best_model.ss_model.linmod.A.weight)<-alpha)

        # Post-processing
        root_name = os.getcwd()
        strPathWeights = os.path.join(root_name, 'weights')
        strPathFigures = os.path.join(root_name, 'figures')
        strMatFileName, strLossFigName, strSimFigName = setSavingName('sn_dd', 
                                nh, n_layers, seq_len, batch_size, lr, strOptimizer, num_epochs, float(lmi_final_tol.gamma.detach().numpy()))
        # Losses
        fig_loss = plt.figure()
        plt.plot(np.log10(dictRes['Loss_hist']), label = 'Train loss')
        plt.plot(np.log10(dictRes['Val_loss_hist']), label = 'Test loss')
        plt.legend()
        fig_loss.savefig(os.path.join(strPathFigures,strLossFigName))

        # Weights
        strDirSaveName = os.path.join(strPathWeights, strMatFileName)
        save_weights(dictRes['weights'], dictRes['biases'], strDirSaveName)

        # Sim
        fig_sim = plot_yTrue_vs_error(y_test, ySim=dictRes['y_sim'])
        fig_sim.savefig(os.path.join(strPathFigures,strSimFigName))

        loss = dictRes['best_loss']

        '''
        Closed-loop simulation
        '''

        # Give the name of the matfile to python function

        net_dims = [nu-nd, nd, nx]
        str_clr_sim_fig_name = strSimFigName[:-4]
        sim_closed_loop(strDirSaveName, net_dims, ts, str_clr_sim_fig_name)


