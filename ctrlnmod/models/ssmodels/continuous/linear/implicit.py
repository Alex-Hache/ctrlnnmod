import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch import Tensor
import os


class ImplicitNNLinear(nn.Module):
    """
        Neural network module corresponding to a (semi) implicit linear state-space model
            Ex^+ = Fx + Bu
            y = Cx
            with E being invertible. In this case no need to solve an algebraic equation just inverse
            E to simulate.
    """

    def __init__(self, input_dim: int, output_dim: int, nx, strSavpath=os.getcwd()) -> None:
        super(ImplicitNNLinear, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim

        self.E = Parameter(torch.Tensor(torch.randn(self.nx, self.nx)))
        self.F = nn.Linear(self.nx, self.nx, bias=False)
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        dx = (self.F(x) + self.B(u)) @ torch.inverse(self.E).T
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.F.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        self.E = Parameter(torch.eye(self.nx, self.nx))
        if is_grad is False:
            self.F.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)
            self.E.requires_grad_(False)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True


class ParamImplicitNNLinear(nn.Module):
    """
        Neural network module corresponding to a parametrized (semi) implicit discrete time
        linear state-space model
            Ex^+ = Fx + Bu
            y = Cx
            with E being invertible. In this case no need to solve an algebraic equation just inverse
            E to simulate.
            E is parameterized to satisfy LMI (18) in this paper : https://arxiv.org/pdf/2004.05290 (Revay et al. 2020)
    """

    def __init__(self, input_dim: int, output_dim: int, nx, strSavpath=os.getcwd()) -> None:
        super(ParamImplicitNNLinear, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim

        self.X = Parameter(torch.Tensor(torch.randn(2 * self.nx, 2 * self.nx)))
        self.Y = Parameter(torch.Tensor(torch.randn(self.nx, self.nx)))
        self.epsilon = 1e-4
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        nx = self.nx
        H = self.X @ self.X.T + self.epsilon * torch.eye(2 * self.nx)
        H11 = H[:nx, :nx]
        P = H[nx:, nx:]
        F = H[nx:, :nx]
        E = 0.5 * (H11 + P + self.Y - self.Y.T)
        dx = x @ (torch.inverse(E) @ F).T + \
            u @ (torch.inverse(E) @ self.B.weight).T
        y = self.C(x)

        return dx, y

    def right_inverse(self, F0: Tensor, B0: Tensor, C0: Tensor, E0: Tensor, is_grad=True):
        '''
            For given (F0, B0, C0, E0) we find a right inverse for X and Y :

        '''
        raise NotImplementedError("Right inverse to Implement")
        self.X.weight = nn.parameter.Parameter(F0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.X.requires_grad_(False)
            self.Y.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True


class ParamImplicitNNLinearAlph(nn.Module):
    r"""
        Neural network module corresponding to a parametrized (semi) implicit discrete time
        linear state-space model with a prescribed lyapunov exponent alpha \in [0,1)
            Ex^+ = Fx + Bu
            y = Cx
            with E being invertible. In this case no need to solve an algebraic equation just inverse
            E to simulate.
            E is parameterized to satisfy LMI (18) in this paper : https://arxiv.org/pdf/2004.05290 (Revay et al. 2020)
    """
    def __init__(self, input_dim: int, output_dim: int, nx, alpha: float = 1.0, strSavpath=os.getcwd()) -> None:
        super(ParamImplicitNNLinearAlph, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim
        self.alpha = alpha

        self.X = Parameter(torch.Tensor(torch.randn(2 * self.nx, 2 * self.nx)))
        self.Y = Parameter(torch.Tensor(torch.randn(self.nx, self.nx)))
        self.epsilon = 1e-4
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        nx = self.nx
        H = self.X @ self.X.T + self.epsilon * torch.eye(2 * self.nx)
        H11 = H[:nx, :nx]
        P = H[nx:, nx:]
        F = H[nx:, :nx]
        E = 0.5 * (H11 + 1 / (self.alpha**2) * P + self.Y - self.Y.T)
        dx = x @ (torch.inverse(E) @ F).T + \
            u @ (torch.inverse(E) @ self.B.weight).T
        y = self.C(x)

        return dx, y

    def right_inverse(self, A0, B0, C0, is_grad=True):
        raise NotImplementedError("Right inverse to Implement")
        self.F.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True
