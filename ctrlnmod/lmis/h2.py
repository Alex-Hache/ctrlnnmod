import torch
from torch import Tensor
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.trace import trace
from cvxpy.error import SolverError
from .base import LMI
from typing import Union, Tuple
import numpy as np
'''
    Every file in the submodule implement a continuous and a discrete version of the lmi.
'''


class H2Cont(LMI):
    r"""
    This LMI gives an upper bound on the H2 norm of continiuous-time linear system.

    attributes
    ------
        * A : Tensor
            state transition matrix
        * B : Tensor
            input matrix
        * C : Tensor
            output matrix
        * gamma2 :
            upper bound on L2 gain
        * P :
            Lyapunov certificate

    methods
    -------

    solve : classmethod
        solve the LMI for given (A,B,C) triplet

    raises
    ------
        ValueError :
            if the SDP problem problem is infeasible or unbounded

    """
    def __init__(self, A: Tensor, B: Tensor, C: Tensor, gamma2: Union[Tensor, None] = None,
                 P: Union[Tensor, None] = None) -> None:
        super(H2Cont, self).__init__()
        self.A = A
        self.B = B
        self.C = C

        # Shapes
        nu, nx, ny = B.shape[1], A.shape[0], C.shape[0]
        self.nu, self.ny, self.nx = nu, ny, nx

        self.shape = nx  # LMI total size
        if gamma2 is not None and P is not None:
            self.gamma2 = gamma2
            self.P = P
        elif gamma2 is not None and P is None:
            _, gamma2_lmi, P = H2Cont.solve(self.A, self.B, self.C)
            if gamma2 > gamma2_lmi:  # Case where the prescribed gamma2 is higher than the minimal one
                self.gamma2 = gamma2
                self.P = P
            else:
                raise ValueError(f"Given gamma2 : {gamma2} is not feasible lowest gamma2 found {gamma2_lmi}")
        else:
            _, gamma2_lmi, P = H2Cont.solve(self.A, self.B, self.C)
            self.gamma2 = gamma2_lmi
            self.P = P.requires_grad_(True)

    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, solver="MOSEK", epsilon=1e-5) -> Tuple[Tensor, Tensor, Tensor]:

        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()

        nx = A.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        Y = Variable((nx, nx), "Y", PSD=True)
        M = A.T @ P + P @ A

        constraints = [(M + C.T @ C) == -Y, P - (epsilon) * np.eye(nx) >> 0]  # type: ignore
        objective = Minimize(trace(Y))
        prob = Problem(objective, constraints=constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()  # If MOSEK is not installed then try SCS by default
        if prob.status not in ["infeasible", "unbounded"]:
            gamma2_lmi = np.sqrt(np.trace(B.T @ P.value @ B))
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        return Tensor(M.value), Tensor([gamma2_lmi]), Tensor(P.value)  # type: ignore

    def forward(self):
        '''
            This not the true H2 just Lyapunov equation. Enforcing it to be
            equal to BB^T is not possible via logdet.
        '''
        M = torch.matmul(self.A.T, self.P) + torch.matmul(self.P, self.A)
        return -M


class H2Disc(LMI):
    r"""
    This LMI gives an upper bound on the H2 norm of continiuous-time linear system.

    attributes
    ------
        * A : Tensor
            state transition matrix
        * B : Tensor
            input matrix
        * C : Tensor
            output matrix
        * gamma2 :
            upper bound on L2 gain
        * P :
            Lyapunov certificate

    methods
    -------

    solve : classmethod
        solve the LMI for given (A,B,C) triplet

    raises
    ------
        ValueError :
            if the SDP problem problem is infeasible or unbounded

    """

    def __init__(self, A: Tensor, B: Tensor, C: Tensor, gamma2: Union[Tensor, None] = None,
                 P: Union[Tensor, None] = None) -> None:
        super(H2Disc, self).__init__()
        self.A = A
        self.B = B
        self.C = C

        # Shapes
        nu, nx, ny = B.shape[1], A.shape[0], C.shape[0]
        self.nu, self.ny, self.nx = nu, ny, nx

        self.shape = nx  # LMI total size
        if gamma2 is not None and P is not None:
            self.gamma2 = gamma2
            self.P = P
        elif gamma2 is not None and P is None:
            _, gamma2_lmi, P = H2Disc.solve(self.A, self.B, self.C)
            if gamma2 > gamma2_lmi:  # Case where the prescribed gamma2 is higher than the minimal one
                self.gamma2 = gamma2
                self.P = P
            else:
                raise ValueError(f"Given gamma2 : {gamma2} is not feasible lowest gamma2 found {gamma2_lmi}")
        else:
            _, gamma2_lmi, P = H2Disc.solve(self.A, self.B, self.C)
            self.gamma2 = gamma2_lmi
            self.P = P.requires_grad_(True)

    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor, solver="MOSEK", epsilon=1e-6) -> Tuple[Tensor, Tensor, Tensor]:

        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()

        nx = A.shape[0]

        P = Variable((nx, nx), "P", PSD=True)
        Y = Variable((nx, nx), "Y", PSD=True)
        M = A.T @ P @ A - P

        constraints = [(M + C.T @ C) == -Y, P - (epsilon) * np.eye(nx) >> 0]  # type: ignore
        objective = Minimize(trace(Y))
        prob = Problem(objective, constraints=constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()  # If MOSEK is not installed then try SCS by default

        if prob.status not in ["infeasible", "unbounded"]:
            gamma2_lmi = np.sqrt(np.trace(B.T @ P.value @ B))
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        return Tensor(M.value), Tensor([gamma2_lmi]), Tensor(P.value)  # type: ignore

    def forward(self):
        '''
            This not the true H2 just Lyapunov equation. Enforcing it to be
            equal to BB^T is not possible via logdet.
        '''
        M = self.A.T @  self.P @ self.A - self.P
        return -M
