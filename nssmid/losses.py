import torch
from torch.nn import MSELoss
from nssmid.layers import DDLayer, DDLayerv2
from nssmid.lmis import *
from nssmid.linalg_utils import *
import torch.nn as nn

def getLoss(config, model = None):
    if config.loss == 'mse':

        if config.lmi is not '':

            if config.lmi== 'lipschitz':
                lmi = Lipschitz(model, config.slope, config.epsilon, config.gamma)
            elif config.lmi == 'lyap':
                A= model.linmod.A.weight
                lmi = LMI_decay_rate(config.alpha_lyap, A, config.epsilon)
            if config.reg_lmi == 'logdet':
                crition = Mixed_LOSS_LMI(lmi, config.mu)
            elif config.reg_lmi == 'dd':
                crition = Mixed_LOSS_LMI_DD(lmi, config.mu, config.bReqGradDD)
            elif config.reg_lmi == 'dd2':
                crition = Mixed_LOSS_LMI_DDv2(lmi, config.mu)
            else:
                raise(NotImplementedError("Please specify a regterm"))
        else:
            crition = MSELoss()
    else:
        raise(NotImplementedError("Only mse loss is implemented so far"))

    return crition

def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal
    
class Mixed_MSELOSS(torch.nn.Module):
    """
        Introduced a convex mixed mse on the state and the ouput
    """
    def __init__(self, alpha =0.5) -> None:
        super(Mixed_MSELOSS, self).__init__()

        self.crit = MSELoss()
        self.alpha = alpha

    def forward(self,y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        return self.alpha*x_mse + (1- self.alpha)*y_mse

class Mixed_MSELOSS_LMI(torch.nn.Module):
    def __init__(self, lmi, alpha= 0.5, mu = 1) -> None:
        super(Mixed_MSELOSS_LMI, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu
        self.alpha = alpha

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.alpha*x_mse + (1- self.alpha)*y_mse
        lmi = self.lmi()
        eig_val,_ = torch.linalg.eig(lmi)
        assert torch.all(torch.real(eig_val)>0)
        L2 = -torch.logdet(lmi)
        return L1 + self.mu*L2

    def update_mu_(self, scale):
        self.mu = self.mu*scale


class Mixed_MSE_LOSS_LMI_DD(torch.nn.Module):
    def __init__(self, lmi, crit = MSELoss(), alpha= 0.5, mu = 1, bRequireGrad = False) -> None:
        super(Mixed_MSE_LOSS_LMI_DD, self).__init__()
        self.crit = crit
        self.lmi = lmi
        self.mu = mu
        self.alpha = alpha
        M = lmi()
        Ui = torch.eye(M.shape[0]) # Shape of the LMI
        self.layer = DDLayer(Ui, bRequires_grad=bRequireGrad)
            

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.alpha*x_mse + (1- self.alpha)*y_mse
        lmi = self.lmi()
        eig_val,_ = torch.linalg.eig(lmi)
        assert torch.all(torch.real(eig_val)>0)

        # DD+ approximation
        dQ = self.layer(lmi)
        L2 = torch.max(dQ)
        return L1 + self.mu*L2

    def update_basis_(self, M):
        self.layer.updateU_(M) #The basis update for basis pursuit.


class Mixed_LOSS_LMI_DD(torch.nn.Module):
    def __init__(self, lmi, mu = 1,bRequires_grad=False) -> None:
        super(Mixed_LOSS_LMI_DD, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu
        lmis = lmi()
        self.ddLayers = nn.ModuleList([DDLayer(torch.eye(lmi.shape[0]),bRequires_grad=bRequires_grad) for lmi in lmis])

    def forward(self, pred, true):

        L1 = self.crit(pred, true)
        lmis = self.lmi()
        # DD+ approximation
        dQ = []
        for i, lmi in enumerate(lmis):
            dQ.append(self.ddLayers[i](lmi))

        return L1, dQ, lmis #objective, dQ, M M being always the first item

    def update_basis_(self, lmis):
        for i, lmi in enumerate(lmis):
            self.ddLayers[i].updateU_(lmi) #The basis update for basis pursuit.


class Mixed_LOSS_LMI_DDv2(torch.nn.Module):
    def __init__(self, lmi, mu = 1,bRequires_grad=False) -> None:
        super(Mixed_LOSS_LMI_DDv2, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu
        lmis = lmi()
        self.ddLayers = nn.ModuleList([DDLayerv2(torch.eye(lmi.shape[0])) for lmi in lmis])

    def forward(self, pred, true):

        L1 = self.crit(pred, true)
        lmis = self.lmi()

        '''
        Si il exist Z tel que -Z <= M-diag(M) <=Z élément par élément
        et diag(M)>= sum(Z,2) alors M est dd+
        '''
        dQ = []
        for i, lmi in enumerate(lmis):
            dQ.append(self.ddLayers[i](lmi))
        return L1, dQ, lmis

    def update_basis_(self, lmis):
        for i, lmi in enumerate(lmis):
            self.ddLayers[i].updateU_(lmi) #The basis update for basis pursuit.


class Mixed_LOSS_LMI(torch.nn.Module):
    def __init__(self, lmi, mu = 1) -> None:
        super(Mixed_LOSS_LMI, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu

    def forward(self, pred, true):

        L1 = self.crit(pred, true)
        lmis = self.lmi()
        #assert torch.all(torch.real(eig_val)>0)
        L2 = 0.0
        for lmi in lmis:
            L2 = L2 -torch.logdet(lmi)
        return L1, L2

    def update_mu_(self, scale):
        self.mu = self.mu*scale
class Mix_MSE_DistAtt(torch.nn.Module):
    def __init__(self, model, alpha=0, gamma = 1) -> None:
        super(Mix_MSE_DistAtt, self).__init__()
        self.crit = MSELoss()
        self.gamma = gamma
        self.alpha = alpha
        self.model = model
    
    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.alpha*x_mse + (1- self.alpha)*y_mse

        # Add L-1 regularization on the distrubance indices
        nu = self.model.input_dim

        #Bu = self.model.linmod.B.weight[:,:nu]
        reg_d, _ = torch.max(torch.abs(self.model.linmod.B.weight),dim=1)
        #reg_u = torch.max(torch.abs(Bu),dim=1)
        reg = reg_d[nu:] #+ reg_u

        return L1 + self.gamma*reg