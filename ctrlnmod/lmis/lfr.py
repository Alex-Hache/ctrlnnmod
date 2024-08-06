import torch
from torch import nn, Tensor
from torch.nn import Parameter
from cvxpy.error import SolverError
from cvxpy import Variable, Problem, Minimize, bmat, diag, trace
from typing import Union, Tuple, Optional, Callable
from .base import LMI
import numpy as np

class AbsoluteStableLFT(LMI):
    def __init__(self, model: Optional[nn.Module] = None, 
                 extract_lmi_matrices: Optional[Callable] = None,
                 A: Optional[Tensor] = None, 
                 B1: Optional[Tensor] = None, 
                 C1: Optional[Tensor] = None, 
                 D11: Optional[Tensor] = None,
                 Lambda_vec: Optional[Tensor] = None, 
                 P: Optional[Tensor] = None,
                 alpha: Tensor = torch.zeros((1)), 
                 mu: Tensor = torch.Tensor([1.0])) -> None:
        super(AbsoluteStableLFT, self).__init__()
        
        self.model = model
        self.extract_lmi_matrices = extract_lmi_matrices
        self.alpha = alpha
        self.mu = mu

        if model is not None and extract_lmi_matrices is not None:
            # Initialize matrices from the model
            self.update_matrices(model, None)
            self.hook = model.register_forward_pre_hook(self.update_matrices)
        else:
            self.A, self.B1, self.C1, self.D11 = A, B1, C1, D11

        if self.A is not None:
            self._initialize_parameters(Lambda_vec, P)
        else:
            raise ValueError("Either provide a model and extract_lmi_matrices, or provide A, B1, C1, D11 matrices.")

    def _initialize_parameters(self, Lambda_vec: Optional[Tensor] = None, P: Optional[Tensor] = None):
        self.nq, self.nx = self.B1.shape[1], self.A.shape[0]
        self.shape = self.nq + self.nx
        
        if self.D11 is None:
            self.D11 = torch.zeros((self.nq, self.nq))

        if Lambda_vec is not None and P is not None:
            self.Lambda_vec = Parameter(Lambda_vec.requires_grad_(True))
            self.P = Parameter(P.requires_grad_(True))
        elif Lambda_vec is not None:
            self.Lambda_vec = Parameter(Lambda_vec.requires_grad_(True))
            _, Lambda, P = self.solve(self.A, self.B1, self.C1, self.D11, self.alpha, self.mu, Lambda=torch.diag(Lambda_vec))
            self.P = Parameter(P.requires_grad_(True))
        else:
            _, Lambda, P = self.solve(self.A, self.B1, self.C1, self.D11, self.alpha, self.mu)
            self.Lambda_vec = Parameter(torch.diag(Lambda).requires_grad_(True))
            self.P = Parameter(P.requires_grad_(True))

    def update_matrices(self, module, input):
        if self.extract_lmi_matrices is None:
            raise ValueError("extract_lmi_matrices is not defined")
        
        matrices = self.extract_lmi_matrices()
        self.A = matrices['A']
        self.B1 = matrices['B1']
        self.C1 = matrices['C1']
        self.D11 = matrices['D11']

        # Re-initialize parameters only if they haven't been initialized yet
        if not hasattr(self, 'Lambda_vec') or not hasattr(self, 'P'):
            self._initialize_parameters()

    @classmethod
    def solve(cls, A: Tensor, B1: Tensor, C1: Tensor, D11: Tensor, alpha: Tensor,
              mu: Tensor = Tensor([1.0]), solver="MOSEK", tol=1e-6, Lambda=None) -> Tuple[Tensor, Tensor, Tensor]:

        A = A.detach().numpy().astype(np.float64)
        B1 = B1.detach().numpy().astype(np.float64)
        C1 = C1.detach().numpy().astype(np.float64)
        D11 = D11.detach().numpy().astype(np.float64)
        mu = mu.detach().numpy().astype(np.float64)
        alpha = alpha.numpy().astype(np.float64)

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
        objective = Minimize(trace(P))  # Feasibility problem

        prob = Problem(objective, constraints=constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()  # If MOSEK is not installed then try SCS by default

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("SDP problem is infeasible or unbounded")

        return Tensor(M.value), Tensor(Lambda.value), Tensor(P.value)

    def _proj(self):
        return 0.5 * (self.P + self.P.T), torch.diag(self.Lambda_vec)
    
    def forward(self):
        P_sym, Lambda = self._proj()
        M11 = self.A.T @ P_sym + P_sym @ self.A + 2 * self.alpha * P_sym
        M12 = P_sym @ self.B1 + self.C1.T @ Lambda
        M22 = -2 / self.mu * Lambda + self.D11.T @ Lambda + Lambda @ self.D11

        M1 = torch.cat([M11, M12], dim=1)
        M2 = torch.cat([M12.T, M22], dim=1)
        M = torch.cat([M1, M2], dim=0)
        return -M, P_sym, Lambda  # Always return a positive definite version