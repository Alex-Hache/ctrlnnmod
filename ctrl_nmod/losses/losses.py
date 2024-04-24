import torch
from torch.nn import MSELoss, Module, ModuleList
from .regularizations import StateRegMSE, _Regularization
from typing import Union, List
from ctrl_nmod.utils.misc import find_module


class _RegularizedLoss(Module):
    '''
        This is base class of regularized loss
    '''

    def __init__(self, regs: Union[ModuleList, None] = None) -> None:
        super().__init__()
        if regs is not None:
            self.regs = regs
        else:
            self.regs = ModuleList([])

    def update(self):
        for reg in self.regs:
            if hasattr(reg, '_update'):
                reg._update()

    def append(self, reg: _Regularization):
        self.regs.append(reg)

    def pop(self, idx: Union[int, slice]):
        self.regs.pop(idx)

    def get_weights(self) -> List[float]:
        weights = []
        for reg in self.regs:
            if hasattr(reg, 'get_weight'):
                weights.append(reg.get_weight())

        return weights

    def get_scalers(self) -> List[float]:
        scalers = []
        for reg in self.regs:
            if hasattr(reg, 'get_scaler'):
                scalers.append(reg.get_scaler())

        return scalers


class MixedMSELoss(_RegularizedLoss):
    r"""
        This loss computes a convex combination between x (state) and y(output)
        \mathcal{L} = (y- \hat{y})^2 + alpha*(x - \hat{x})
    """

    def __init__(self, alpha: float, scale=1.0) -> None:
        regs = ModuleList([StateRegMSE(alpha=alpha, scale=scale)])
        super().__init__(regs)
        self.crit = MSELoss()
        self.alpha = alpha
        self.scale = scale

    def forward(self, y_true, y_sim, x_true, x_sim):
        y_mse = self.crit(y_true, y_sim)
        reg_loss = torch.zeros((1))
        for i, regularization in enumerate(self.regs):
            if i == 0:  # First index is for state_regularization
                reg_loss += regularization(x_true, x_sim)
            else:
                reg_loss += regularization()
        return y_mse + reg_loss

    def __repr__(self):
        return f"Mixed MSE : alpha = {self.alpha}"

    def update(self) -> None:
        super().update()
        self.alpha = self.regs[0].get_weight()


class MixedNMSEReg(MixedMSELoss):
    '''
        This is the same verion than the regular MSE but normalized by the
        mean of y_true**2. This loss has values ranging form -Inf to 1
    '''

    def __init__(self, alpha: float, scale=1.0) -> None:
        super().__init__(alpha=alpha, scale=scale)

    def forward(self, y_true, y_sim, x_true, x_sim):
        y_mse = self.crit(y_true, y_sim)
        y_nmse = y_mse / (torch.mean(y_true**2))
        reg_x = find_module(self.regs, StateRegMSE)
        if reg_x is not None:
            x_mse = reg_x(x_true, x_sim)
            x_nmse = x_mse / (torch.mean(x_true**2))
        else:
            raise ValueError("Module not found")
        return 0.5 * (y_nmse + x_nmse)

    def __repr__(self):
        return f"Mixed NMSE : alpha = {self.alpha} \n" + f"Regs : {self.regs}"


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
