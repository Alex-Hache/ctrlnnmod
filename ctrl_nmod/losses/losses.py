import torch
from torch.nn import MSELoss

class Mixed_MSELOSS(torch.nn.Module):
    """
        Introduced a convex mixed mse on the state and the ouput
    """
    def __init__(self, alpha=0.5) -> None:
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