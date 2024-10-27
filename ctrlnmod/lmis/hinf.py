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

class HInfBase(LMI):
    def __init__(self,  model: Optional[torch.nn.Module] = None, 
                 extract_lmi_matrices: Optional[Callable] = None,
                 A: Optional[Tensor] = None, 
                 B: Optional[Tensor] = None, 
                 C: Optional[Tensor] = None, 
                 D: Optional[Tensor] = None,
                 gamma: Optional[Tensor] = None, 
                 P: Optional[Tensor] = None,
                 alpha: Tensor = torch.zeros((1))) -> None:
        super(HInfBase, self).__init__()


        self.model = model
        self.extract_lmi_matrices = extract_lmi_matrices

        if model is not None and extract_lmi_matrices is not None:
            self.update_matrices()
            self.hook = self.register_forward_pre_hook(self.update_matrices())
        else: 
            self.A = A
            self.B = B
            self.C = C
            self.D = D if D is not None else torch.zeros((self.ny, self.nu))

            self.nu, self.nx, self.ny = B.shape[1], A.shape[0], C.shape[0]


        self.shape = self.nu + self.ny + self.nx
        self.alpha = alpha

        if gamma is not None and P is not None:
            self.gamma = gamma
            self.P = P
        elif gamma is not None and P is None:
            _, obj, vars = self.solve(self.A, self.B, self.C, self.D, self.alpha)
            if gamma > vars['gamma']:
                self.gamma = gamma
                self.P = vars['P']
            else:
                raise ValueError(f"Given gamma: {gamma} is not feasible. Lowest gamma found: {vars['gamma']}")
        else:
            _, obj, vars = self.solve(self.A, self.B, self.C, self.D, self.alpha)
            self.gamma = vars['gamma']
            self.P = vars['P'].requires_grad_(True)

    
    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, D: Tensor, alpha: Tensor,
              solver: str = "MOSEK", tol: float = 1e-6) -> Tuple[np.ndarray, float, dict]:
        raise NotImplementedError("Subclasses must implement this method")

    def _symP(self) -> Tensor:
        return 0.5 * (self.P + self.P.T)

    def forward(self) -> Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class HInfCont(HInfBase):
    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, D: Tensor, alpha: Tensor,
              solver: str = "MOSEK", tol: float = 1e-6) -> Tuple[np.ndarray, float, dict]:
        A, B, C, D = [x.detach().numpy() for x in [A, B, C, D]]
        nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        gam = Variable()
        M = bmat([
            [A.T @ P + P @ A, P @ B, C.T],
            [B.T @ P, -gam * np.eye(nu), D.T],
            [C, D, -gam * np.eye(ny)]
        ])

        constraints: List[Union[bool, np.ndarray]] = [
            M << -tol * np.eye(nx + nu + ny),
            P - tol * np.eye(nx) >> 0,
            A.T @ P + P @ A + 2 * alpha * P << -tol * np.eye(nx),
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

        return M.value, prob.value, gam.value, P.value

    def forward(self) -> Tensor:
        P = self._symP()
        M11 = self.A.T @ P + P @ self.A + 1 / self.gamma * self.C.T @ self.C
        M12 = P @ self.B
        M22 = -self.gamma * torch.eye(self.nu, device=self.A.device)

        M = torch.cat((torch.cat((M11, M12), 1), torch.cat((M12.T, M22), 1)), 0)
        return -M

    def update_matrices(self, *args):
        if self.extract_lmi_matrices is None:
            raise ValueError("extract_lmi_matrices is not defined")
        
        A, B, C, D = self.extract_lmi_matrices()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    
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