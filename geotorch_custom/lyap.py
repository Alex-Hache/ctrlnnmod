import torch
from torch import Tensor
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.variable import Variable
from cvxpy import trace
import numpy as np
from cvxpy.error import SolverError
from geotorch_custom.product import ProductManifold
from geotorch_custom.psd import PSD
from geotorch_custom.skew import Skew
from geotorch_custom.exceptions import (
    VectorError,
    NonSquareError,
    InManifoldError,
)
from geotorch_custom.utils import _extra_repr


def get_lyap_exp(A):
    return -torch.max(torch.real(torch.linalg.eigvals(A)))


class AlphaStable(ProductManifold):
    def __init__(self, size, alpha: float, triv="expm"):

        n, tensorial_size = AlphaStable.parse_size(size)
        super().__init__(AlphaStable.manifolds(n, tensorial_size, triv))
        self.n = n
        self.tensorial_size = tensorial_size
        self.alpha = alpha
        self.In = torch.eye(n)

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        tensorial_size = size[:-2]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n, tensorial_size

    @staticmethod
    def manifolds(n, tensorial_size, triv):
        size_p = tensorial_size + (n, n)
        size_q = tensorial_size + (n, n)
        return PSD(size_q, triv=triv), PSD(size_p, triv=triv), Skew()

    def submersion(self, Q, P, S):
        return P @ (-0.5 * Q + S) - self.alpha * self.In

    def forward(self, X1, X2, X3):
        Q, P, S = super().forward([X1, X2, X3])
        return self.submersion(Q, P, S)

    def submersion_inv(self, A, check_in_manifold=True, epsilon=1e-2, solver="MOSEK"):
        if check_in_manifold and not self.in_manifold_eigen(A, epsilon):
            print(f"Enforced alpha = {self.alpha} and given alpha = {get_lyap_exp(A)}")
            raise InManifoldError(get_lyap_exp(A), self.alpha)
        with torch.no_grad():
            A = A.detach().numpy()
            nx = A.shape[0]
            P = Variable((nx, nx), "P", PSD=True)
            Q = (
                A.T @ P + P @ A + 2 * self.alpha * P  # type: ignore
            )  # solve the negative definite version
            constraints = [Q << -epsilon * np.eye(nx), P - (epsilon) * np.eye(nx) >> 0]  # type: ignore
            objective = Minimize(trace(P))  # Feasibility problem

            prob = Problem(objective, constraints=constraints)
            try:
                prob.solve(solver)
            except SolverError:
                prob.solve()  # If MOSEK is not installed then try SCS by default

            if prob.status  in ["infeasible", "unbounded"]:
                raise ValueError("SDP problem is infeasible or unbounded")

            # Now initialize
            P_inv = torch.inverse(Tensor(P.value))
            Q = Tensor(-Q.value)

            S = Tensor(P.value) @ Tensor(A) + 0.5 * Q + self.alpha * Tensor(P.value)
            print(torch.real(torch.linalg.eigvals(Q)))
        return Q, P_inv, S

    def right_inverse(self, A, check_in_manifold=True):
        Q, P, S = self.submersion_inv(A, check_in_manifold)
        X1, X2, X3 = super().right_inverse([Q, P, S])
        return X1, X2, X3

    def in_manifold_eigen(self, A, eps=1e-6):
        r"""
        Check that all eigenvalues are lower than -alpha
        """
        if A.size()[:-2] != self.tensorial_size:  # check dimensions
            return False
        else:
            eig = torch.linalg.eigvals(A)
            reig = torch.real(eig)
            return (reig <= -self.alpha + eps).all().item()

    def sample(self, init_=torch.nn.init.xavier_normal_):
        with torch.no_grad():
            X_p = torch.empty(*(self.tensorial_size + (self.n, self.n)))
            init_(X_p)
            X_q = torch.empty_like(X_p)
            init_(X_q)
            X_s = torch.empty_like(X_p)
            init_(X_s)
            P = X_p @ X_p.transpose(-2, -1)
            Q = X_q @ X_q.transpose(-2, -1)
            S = X_s - X_s.transpose(-2, -1)

            return P @ (-0.5 * Q + S) - self.alpha * self.In

    def extra_repr(self) -> str:
        return _extra_repr(
            n=self.n,
            alpha=self.alpha,
            tensorial_size=self.tensorial_size,
        )
