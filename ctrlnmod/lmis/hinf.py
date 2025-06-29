import torch
from torch import Tensor
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.error import SolverError
from .base import LMI
from typing import Union, Tuple, Optional, List, Callable
import numpy as np
import math
from abc import abstractmethod


class HInfBase(LMI):
    ExtractMatricesFn = Callable[[], Tuple[Tensor, ...]]
    def __init__(self,  module: torch.nn.Module,
                 extract_matrices: ExtractMatricesFn,
                 solver: str = 'MOSEK'
                 ) -> None:
        super(HInfBase, self).__init__(module, extract_matrices)


        self.P = None
        self.gamma = None

        # Dimensions
        self.nu = None
        self.ny = None
        self.nx = None

    def _symP(self) -> Tensor:
        return 0.5 * (self.P + self.P.T)

    
class HInfCont(HInfBase):
    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, D: Tensor, alpha: float = 0.0,
              solver: str = "MOSEK", tol: float = 1e-8) -> Tuple[Tensor, float, Tensor]:
        A, B, C, D = [x.detach().numpy() for x in [A, B, C, D]]
        nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        gam = Variable()
        M = bmat([
            [A.T @ P + P @ A + C.T @ C, P @ B],
            [B.T @ P, -gam * np.eye(nu)]
        ])

        constraints: List[Union[bool, np.ndarray]] = [
            M << -tol * np.eye(nx + nu),
            P - tol * np.eye(nx) >> 0,
            gam - tol >= 0,
        ]
        objective = Minimize(gam)

        prob = Problem(objective, constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("SDP problem is infeasible or unbounded")

        return -torch.Tensor(M.value), math.sqrt(prob.value), Tensor(P.value)

    def forward(self) -> Tuple[Tensor, ...]:
        P = self._symP()
        M11 = self.A.T @ P + P @ self.A + self.C.T @ self.C
        M12 = P @ self.B
        M22 = -self.gamma**2 * torch.eye(self.nu, device=self.A.device)

        M = torch.cat((torch.cat((M11, M12), 1), torch.cat((M12.T, M22), 1)), 0)
        return -M, P

    def _update_matrices(self, *args) -> None:
            A, B, C = self.extract_matrices()
            self.A = A
            self.B = B
            self.C = C

            if self.P is None or self.gamma is None:
                self.init_(self.A, self.B, self.C)

    def init_(self,
                A: Optional[Tensor] = None,
                B: Optional[Tensor] = None,
                C: Optional[Tensor] = None,
                epsilon: float = 1e-4,
                solver: str = 'MOSEK'):
            
        if (A is None and B is None and C is None):
                A, B, C = self.extract_matrices()
                
        assert A is not None and B is not None and C is not None 

        self.nx = B.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]
        
        _, gamma, P = HInfCont.solve(A.detach(), B.detach(), C.detach(), torch.zeros((self.ny, self.nu)),solver=solver, tol=epsilon)

        self.gamma = gamma
        self.P = P
        
    
class HInfDisc(HInfBase):
    def __init__(self, *args, **kwargs):
        super(HInfDisc, self).__init__(*args, **kwargs)
        self.register_buffer('Inu', torch.eye(self.nu))
        self.register_buffer('Iny', torch.eye(self.ny))

    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, D: Tensor, alpha: Tensor,
              solver: str = "MOSEK", tol: float = 1e-6) -> Tuple[np.ndarray, float, dict]:
        A, B, C, D = [x.detach().numpy() for x in [A, B, C, D]]
        nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        gam = Variable()
        M = bmat([
            [A.T @ P @ A - P, A.T @ P @ B, C.T],
            [B.T @ P @ A, B.T @ P @ B - gam * np.eye(nu), D.T],
            [C, D, -gam * np.eye(ny)]
        ])

        constraints: List[Union[bool, np.ndarray]] = [
            M << -tol * np.eye(nx + nu + ny),
            P - tol * np.eye(nx) >> 0,
            A.T @ P @ A - alpha**2 * P << -tol * np.eye(nx),
            gam - tol >= 0,
        ]
        objective = Minimize(gam)

        prob = Problem(objective, constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("SDP problem is infeasible or unbounded")

        return M.value, prob.value,  gam.value, P.value

    def forward(self) -> Tensor:
        P = self._symP()
        M11 = self.A.T @ P @ self.A - P
        M12 = self.A.T @ P @ self.B
        M13 = self.C.T
        M22 = self.B.T @ P @ self.B - self.gamma * self.Inu
        M23 = self.D.T
        M33 = -self.gamma * self.Iny

        M = torch.stack([
            torch.cat([M11, M12, M13], dim=1),
            torch.cat([M12.T, M22, M23], dim=1),
            torch.cat([M13.T, M23.T, M33], dim=1)
        ])
        return -M