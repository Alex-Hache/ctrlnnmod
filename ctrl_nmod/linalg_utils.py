import torch

def isSDP(L):
    '''
    L is tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    isAllEigPos = torch.all(torch.real(eigval)>0) 
    isSymetric = torch.all(L == L.T)
    if not isAllEigPos:
        print("Not all eigenvalues are positive")
    if not isSymetric:
        print("Matrix is not symmetric")
    bSDP = isSymetric and isAllEigPos
    return bSDP

def getEigenvalues(L):
    '''
        params :
            - L pytorch Tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    return eigval

