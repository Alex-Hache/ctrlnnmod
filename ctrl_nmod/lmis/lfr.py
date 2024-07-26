import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from cvxpy.error import SolverError
from cvxpy import Variable, Problem, Minimize, bmat, diag
from typing import Union, Tuple, Optional
# from ..linalg.utils import isSDP
from .base import LMI


class AbsoluteStableLFT(LMI):
    r"""
    This LMI gives an upper bound on the L2 gain of continiuous-time linear system.

    attributes
    ------
        * A : Tensor
        * B1 : Tensor
        * C1 : Tensor
        * D11 : Tensor
        * alpha :
            contraction metric i.e. largest lyapunov exponent
        * mu :
            upper bound on absolute stability
            i.e. maximum admissible slope for the activation functions
            in the case of neural networks
        * P :
            Lyapunov certificate

    methods
    -------

    solve : classmethod
        solve the LMI for given (A,B,C,D11) quadruplet

    raises
    ------
        ValueError :
            if the SDP problem problem is infeasible or unbounded

    """

    def __init__(self, A: Tensor, B1: Tensor, C1: Tensor, D11: Optional[Tensor] = None,
                 Lambda: Union[Tensor, None] = None, P: Union[Tensor, None] = None,
                 alpha: Tensor = torch.zeros((1)), mu: Tensor = torch.Tensor([1.0])) -> None:
        super(AbsoluteStableLFT, self).__init__()
        self.A = A
        self.B1 = B1
        self.C1 = C1
        self.mu = mu

        # Shapes
        nq, nx = B1.shape[1], A.shape[0]
        self.nq, self.nx = nq, nx

        if D11 is not None:
            self.D11 = D11
        else:
            self.D11 = torch.zeros((nq, nq))

        self.shape = nq + nx  # LMI total size
        self.alpha = alpha
        if Lambda is not None and P is not None:
            self.Lambda = Lambda
            self.P = P
        elif Lambda is not None and P is None:
            self.Lambda = Lambda
            _, Lambda, P = AbsoluteStableLFT.solve(self.A, self.B1, self.C1, self.D11, self.alpha, self.mu, Lambda=self.Lambda)
        else:
            _, Lambda, P = AbsoluteStableLFT.solve(self.A, self.B1, self.C1, self.D11, self.alpha, self.mu)
            self.Lambda = Parameter(Lambda.requires_grad_(True))
            self.P = Parameter(P.requires_grad_(True))

    @classmethod
    def solve(cls, A: Tensor, B1: Tensor, C1: Tensor, D11: Tensor, alpha: Tensor,
              mu: Tensor = Tensor([1.0]), solver="MOSEK", tol=1e-6, Lambda=None) -> Tuple[Tensor, Tensor, Tensor]:

        A = A.detach().numpy()
        B1 = B1.detach().numpy()
        C1 = C1.detach().numpy()
        D11 = D11.detach().numpy()
        mu = mu.detach().numpy()

        nx = A.shape[0]
        nq = B1.shape[1]

        P = Variable((nx, nx), "P", PSD=True)
        if Lambda is None:
            Lambda = diag(Variable(nq))
        M = bmat(
            [
                [A.T @ P + P @ A + 2 * alpha * P, P @ B1 + C1.T @ Lambda],
                [B1.T @ P + Lambda @ C1, -2 / mu * Lambda + D11.T @ Lambda + Lambda @ D11],  # type: ignore
            ]
        )
        constraints = [
            M << -tol * np.eye(nx + nq),  # type: ignore
            P - (tol) * np.eye(nx) >> 0,
            Lambda - tol * np.eye(nq) >> 0   # type: ignore,
        ]
        objective = Minimize(0)  # Feasibility problem

        prob = Problem(objective, constraints=constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()  # If MOSEK is not installed then try SCS by default

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("SDP problem is infeasible or unbounded")

        return Tensor(M.value), Tensor(Lambda.value), Tensor(P.value)

    def forward(self):
        M11 = self.A.T @ self.P + self.P @ self.A
        M12 = self.P @ self.B1 + self.C1.T @ self.Lambda
        M22 = -2 / self.mu * self.Lambda + self.D11.T @ self.Lambda + self.Lambda @ self.D11

        M1 = torch.cat([M11, M12], dim=1)
        M2 = torch.cat([M12.T, M22], dim=1)
        M = torch.cat([M1, M2], dim=0)
        return -M
