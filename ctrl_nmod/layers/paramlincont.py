"""
    This module implements several parameterizations for continuous neural state-space models
    it includes :
        - Alpha stable linear models
        - L2 gain bounded linear models -- H inifinty norm
        - QSR dissipative linear models including :
            - L2 gain bounded
        - H2 gain
        - Incrementally QSR dissipative models including :
            - Incremental L2 gain bounded
            - Incrementally input passive
            - Incrementally output passive
"""
from typing import List, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.utils.parametrize as parametrize
import cvxpy as cp
import geotorch as geo


class AlphaStable(Module):
    """
    This function implements alpha matrices i.e.
    they all have neagtive real parts inferior to -alpha
    """

    def __init__(self, n, alpha=1.0, left=True) -> None:
        """
        If left is set as True then X is solution of
        X' @ P^{-1} + P^{-1}@X +2*alpha *P = -Q
        if set as False then X is solution of
        X@P^{-1}+ P^{-1}@X' +2*alpha*P = -Q
        """
        super().__init__()
        self.n = n
        self.alpha = alpha
        self.left = left

        # Registering parameterizations
        self.P = torch.randn((n, n))
        geo.positive_definite(self, "P")

        self.Q = torch.randn((n, n))
        geo.positive_definite(self, "Q")

        self.S = torch.randn((n, n))
        geo.skew(self, "S")

    def forward(self, P, Q, S, alpha):
        if self.left:
            return self.P @ 


class ContinuousLinearSS(Module):
    """
    This is base class for continuous RNN for which weights
    can be written as a linear state-space model
    """

    def __init__(
        self, *args: Union[List[Tensor], List[int]], stable: bool = True
    ) -> None:
        """
        Accept either
            [A, B, C, D] tensors
            or
            [nu, nx, ny] or system dimensions
        """
        super(ContinuousLinearSS, self).__init__()

        if all([isinstance(arg, Tensor) for arg in args]):
            # We define the ss model by its matrices

            A = args[0]
            B = args[1]
            C = args[2]
            D = args[3]

            self.nu = B.shape[1]  # type: ignore
            self.nx = A.shape[0]  # type: ignore
            self.ny = C.shape[0]  # type: ignore

            self.weight = Parameter(
                torch.cat(
                    (torch.cat((A, B), 1), torch.cat((C, D), 1)), 0  # type: ignore
                )
            ).requires_grad_(
                True
            )  # type: ignore
        elif all([isinstance(dim, int) for dim in args]):
            self.nu = args[0]
            self.nx = args[1]
            self.ny = args[2]
            W = torch.rand((self.nx + self.ny, self.nx + self.nu))  # type: ignore

            if stable:
                A = -torch.eye(self.nx)  # type: ignore # Stable intialization
                W[: self.nx, : self.nx] = A
            self.weight = Parameter(W).requires_grad_(True)
        else:
            raise TypeError("Please pass a tensor or dimensions")

    def forward(self, u, x):
        out = torch.cat((x, u), dim=0) @ self.weight.T
        dx, y = out[: self.nx, self.nx :]
        return dx, y

    def eval_(self):
        0
