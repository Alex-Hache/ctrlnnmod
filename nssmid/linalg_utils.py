import torch

def isSDP(L):
    '''
    L is tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    isSDP = torch.all(torch.real(eigval)>0)
    return isSDP

def getEigenvalues(L):
    '''
        params :
            - L pytorch Tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    return eigval

