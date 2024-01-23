import numpy as np
import torch
from scipy.io import savemat

import matlab.engine


def findBLA(u : np.ndarray, y : np.ndarray, nx : int,
            ts : float, model_type : str = 'discrete',
            save : bool = False, strNameSave : str = "BLA.mat"):
    """
        Perform linear identification using scriptName Matlab function

        params : 
            * u : input data
            * y : output data
            * nx : dimension of state-space
            * ts : sample time
            * save : do we want to save the linMod structure into a mat file
        returns :
            A, B, C, D matrices of the identified state-space as pytorch Tensors
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(eng.pwd()))
    data_InMat = matlab.double(u.tolist()) # N_samples x N_channels matlab double
    data_OutMat = matlab.double(y.tolist()) # N_samples x N_channels matlab double
    nx = matlab.double([nx])
    mTs = matlab.double([ts])
    linMod = eng.initLin(data_InMat, data_OutMat, nx, mTs, model_type) # Call the initLin.m script  
    eng.quit()

    if save :
        linMod['A'] = np.asarray(linMod['A'])
        linMod['B'] = np.asarray(linMod['B'])
        linMod['C'] = np.asarray(linMod['C'])
        linMod['uTot'] = np.asarray(u)
        linMod['yTot'] = np.asarray(y)
        savemat(strNameSave, linMod)
    A = torch.from_numpy(np.asarray(linMod['A'])).to(torch.float32)
    B = torch.from_numpy(np.asarray(linMod['B'])).to(torch.float32)
    C = torch.from_numpy(np.asarray(linMod['C'])).to(torch.float32)
    D = torch.from_numpy(np.asarray(linMod['D'])).to(torch.float32).unsqueeze(dim=0)
    return A, B, C, D



def sim_closed_loop(strMatFileWeights, net_dims, dt, strNameSaveFig : str ):
    dt = matlab.double([dt])
    eng = matlab.engine.start_matlab()
    net_dims = matlab.double(net_dims)
    eng.addpath(eng.genpath(eng.pwd()))

    eng.load_workspace(strMatFileWeights, net_dims, dt, strNameSaveFig, nargout=0)
    eng.closedLoopresults_pendulum(nargout=0)
    eng.quit()
