import torch
from torch.nn import MSELoss, Module
from .regularizations import RegularizationsList
from typing import Union


class Mixed_MSELoss(Module):
    r"""
        This loss computes a convex combination between x (state) and y(output)
        \mathcal{L} = \alpha*(y- \hat{y})^2 + (1 - \alpha)*(x - \hat{x})
    """
    def __init__(self, alpha: float = 1) -> None:
        super().__init__()
        self.mse_y = MSELoss()
        self.mse_x = MSELoss()
        self.alpha = alpha

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.mse_x(x_true, x_sim)
        y_mse = self.mse_y(y_true, y_sim)
        return self.alpha * y_mse + (1 - self.alpha) * x_mse


class RegularizedLoss(Module):
    '''
        This is base class of regularized loss
    '''
    def __init__(self, regs: Union[RegularizationsList, None] = None) -> None:
        super().__init__()
        self.regs = regs

    def update(self):
        if self.regs is not None:
            for reg in self.regs.regularizations:
                if hasattr(reg, 'update'):
                    reg.update()  # type: ignore


class MixedMSEReg(RegularizedLoss):
    '''
    '''
    def __init__(self, nu: float, regs: Union[RegularizationsList, None] = None) -> None:
        super().__init__(regs)
        self.nu = nu
        self.crit = MSELoss()

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        if self.regs is not None:
            reg = self.regs()
            return self.nu*y_mse + (1-self.nu)*x_mse + reg
        else:
            return self.nu*y_mse + (1-self.nu)*x_mse

    def __repr__(self):
        return f"Mixed MSE : nu = {self.nu}"


class MixedNMSEReg(RegularizedLoss):
    '''
    '''
    def __init__(self, regs: Union[RegularizationsList, None] = None) -> None:
        super().__init__(regs)
        self.crit = MSELoss()

    def forward(self, y_true, y_sim, x_true, x_sim):
        y_mse = self.crit(y_true, y_sim)
        y_nmse = y_mse/(torch.mean(y_true**2))
        if self.regs is not None:
            reg = self.regs()
            return y_nmse + reg
        else:
            return y_nmse

    def __repr__(self):
        return "Mixed NMSE"


class Mixed_MSE_LMI(MixedMSEReg):
    r"""
        This Loss is a regular mse loss added with a logdet barrier term
        \mathcal{L} = MixedMSE - \mu logdet(lmi)
        where lmi is a Module producing a positive definite matrix
    """
    def __init__(self, lmi: Module, mu: float, alpha: float = 1) -> None:
        super().__init__(alpha)
        self.lmi = lmi
        self.mu = mu

    def forward(self, y_true, y_sim, x_true, x_sim):
        mse_loss = super().forward(y_true, y_sim, x_true, x_sim)
        barrier = torch.logdet(self.lmi())

        return mse_loss - self.mu*barrier

    def update(self, scale) -> None:
        r'''
            This function updates barrier term ponderation term if called
            self.mu = self.mu * scale
        '''
        self.mu = self.mu*scale


class Mixed_MSELOSS_LMI(torch.nn.Module):
    def __init__(self, lmi, alpha=0.5, mu=1) -> None:
        super(Mixed_MSELOSS_LMI, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu
        self.alpha = alpha

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.alpha * x_mse + (1 - self.alpha) * y_mse
        lmi = self.lmi()
        eig_val, _ = torch.linalg.eig(lmi)
        assert torch.all(torch.real(eig_val) > 0)
        L2 = -torch.logdet(lmi)
        return L1 + self.mu * L2

    def update_mu_(self, scale):
        self.mu = self.mu * scale


class Mix_MSE_DistAtt(torch.nn.Module):
    def __init__(self, model, alpha=0, gamma=1) -> None:
        super(Mix_MSE_DistAtt, self).__init__()
        self.crit = MSELoss()
        self.gamma = gamma
        self.alpha = alpha
        self.model = model

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.alpha * x_mse + (1 - self.alpha) * y_mse

        # Add L-1 regularization on the distrubance indices
        nu = self.model.input_dim

        # Bu = self.model.linmod.B.weight[:,:nu]
        reg_d, _ = torch.max(torch.abs(self.model.linmod.B.weight), dim=1)
        # reg_u = torch.max(torch.abs(Bu),dim=1)
        reg = reg_d[nu:]  # + reg_u

        return L1 + self.gamma * reg
