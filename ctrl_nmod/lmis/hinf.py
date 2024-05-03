import torch
from torch import Tensor
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.bmat import bmat

from .base import LMI
from typing import Union, Tuple, Optional
import numpy as np
'''
    Every file in the submodule implement a continuous and a discrete version of the lmi
'''


class HInfCont(LMI):

    def __init__(self, A: Tensor, B: Tensor, C: Tensor, D: Optional[Tensor] = None,
                 gamma: Union[Tensor, None] = None, P: Union[Tensor, None] = None,
                 alpha: Tensor = torch.zeros((1))) -> None:
        super(HInfCont, self).__init__()
        self.A = A
        self.B = B
        self.C = C

        print(self.B.shape)
        print(self.A.shape)
        print(self.C.shape)
        # Shapes
        nu, nx, ny = B.shape[1], A.shape[0], C.shape[0]
        self.nu, self.ny, self.nx = nu, ny, nx

        if D is not None:
            self.D = D
        else:
            self.D = torch.zeros((ny, nu))

        self.shape = nu + ny + nx  # LMI total size
        self.alpha = alpha
        if gamma is not None and P is not None:
            self.gamma = gamma
            self.P = P
        elif gamma is not None and P is None:
            _, gamma_lmi, P = HInfCont.solve(self.A, self.B, self.C, self.D, self.alpha)
            if gamma > gamma_lmi:  # Case where the prescribed gamma is higher than the minimal one
                self.gamma = gamma
                self.P = P
            else:
                raise ValueError(f"Given gamma : {gamma} is not feasible lowest gamma found {gamma_lmi}")
        else:
            _, gamma_lmi, P = HInfCont.solve(self.A, self.B, self.C, self.D, self.alpha)
            self.gamma = gamma_lmi
            self.P = P.requires_grad_(True)

    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, D: Tensor, alpha: Tensor, solver="MOSEK", tol=1e-6) -> Tuple[Tensor, Tensor, Tensor]:

        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()
        D = D.detach().numpy()

        nx = A.shape[0]
        nu = B.shape[1]
        ny = C.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        gam = Variable()
        M = bmat(
            [
                [A.T @ P + P @ A, P @ B, C.T],
                [B.T @ P, -gam * np.eye(nu), D.T],  # type: ignore
                [C, D, -gam * np.eye(ny)]  # type: ignore
            ]
        )
        constraints = [
            M << -tol * np.eye(nx + nu + ny),  # type: ignore
            P - (tol) * np.eye(nx) >> 0,  # type: ignore
            A.T @ P + P @ A + 2 * alpha * P << -(tol * np.eye(nx)),
            gam - tol >= 0,  # type: ignore
        ]
        objective = Minimize(gam)  # Feasibility problem

        prob = Problem(objective, constraints=constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
            gmma_lmi = gam.value
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        return Tensor(M.value), Tensor(np.array([gmma_lmi])), Tensor(P.value)

    def forward(self):
        # TODO implement version with nonzero D
        M11 = torch.matmul(self.A.T, self.P) + torch.matmul(self.P, self.A) + 1 / self.gamma * torch.matmul(self.C.T, self.C)
        M12 = torch.matmul(self.P, self.B)
        M22 = -self.gamma * torch.eye(self.nu)

        M = torch.cat((torch.cat((M11, M12), 1), torch.cat((M12.T, M22), 1)), 0)
        return -M


class HInfDisc(LMI):
    def __init__(self, A: Tensor, B: Tensor, C: Tensor, D: Union[Tensor, None],
                 gamma: Union[Tensor, None], P: Union[Tensor, None],
                 alpha: Tensor = torch.zeros((1))) -> None:
        super(HInfDisc, self).__init__()
        self.A = A
        self.B = B
        self.C = C

        # Shapes
        nu, nx, ny = B.shape[1], A.shape[0], C.shape[0]
        self.nu, self.ny, self.nx = nu, ny, nx

        self.register_buffer('Inu', torch.eye(nu))
        self.register_buffer('Iny', torch.eye(ny))

        if D is not None:
            self.D = D
        else:
            self.D = torch.zeros((ny, nu))

        self.shape = nu + ny + nx  # LMI total size
        self.alpha = alpha
        if gamma is not None and P is not None:
            self.gamma = gamma
            self.P = P
        elif gamma is not None and P is None:
            _, gamma_lmi, P = HInfDisc.solve(self.A, self.B, self.C, self.D, self.alpha)
            if gamma > gamma_lmi:  # Case where the prescribed gamma is higher than the minimal one
                self.gamma = gamma
                self.P = P
            else:
                raise ValueError(f"LMI is not feasible with given gamma : {gamma} -- lowest gamma found {gamma_lmi}")
        else:
            _, gamma_lmi, P = HInfDisc.solve(self.A, self.B, self.C, self.D, self.alpha)
            self.gamma = gamma_lmi
            self.P = P

    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, D: Tensor, alpha: Tensor, solver="MOSEK", tol=1e-6):

        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()
        D = D.detach().numpy()

        nx = A.shape[0]
        nu = B.shape[1]
        ny = C.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        gam = Variable()
        M = bmat(
            [
                [A.T @ P @ A - P, A.T @ P @ B, C.T],
                [B.T @ P @ A, B.T @ P @ B - gam * np.eye(nu), D.T],  # type: ignore
                [C, D, -gam * np.eye(ny)]  # type: ignore
            ]
        )
        constraints = [
            M << -tol * np.eye(nx + nu + ny),  # type: ignore
            P - (tol) * np.eye(nx) >> 0,  # type: ignore
            A.T @ P @ A - alpha**2 * P << -(tol * np.eye(nx)),
            gam - tol >= 0,  # type: ignore
        ]
        objective = Minimize(gam)  # Feasibility problem

        prob = Problem(objective, constraints=constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
            gmma_lmi = gam.value
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        return Tensor(M.value), Tensor([gmma_lmi]), Tensor(P.value)

    def forward(self):
        r'''
         M = [A^TPA-P  A^T*P*B C^T
                * B^T*P*B -gammaI 0(nu,ny)
                *  *  -\gamma I_{nu}] <0
        ** returns : -M
        '''
        M11 = torch.matmul(torch.matmul(self.A.T, self.P), self.A) - self.P
        M12 = torch.matmul(torch.matmul(self.A.T, self.P), self.B)
        M13 = self.C.T
        M22 = torch.matmul(torch.matmul(self.B.T, self.P), self.B) - self.gamma * self.Inu
        M23 = self.D.T
        M33 = self.gamma * self.Iny

        M = torch.Tensor([[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]])
        return -M
