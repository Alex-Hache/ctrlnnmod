from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
import cvxpy as cp
# from ..linalg.utils import isSDP
from .base import LMI


class LyapunovDiscrete(LMI):
    """
    Lyapunov LMI for discrete-time systems.
    """

    def __init__(self, A: Tensor, alpha: float) -> None:
        """
        Initialize the Lyapunov LMI for discrete-time systems.

        Args:
            A (Tensor): The system matrix A.
            alpha (float): The decay rate alpha for alpha-stability.
        """
        super().__init__()
        self.A = Parameter(A)
        self.alpha = alpha
        self.P = Parameter(torch.eye(A.size(0)))

    def forward(self) -> Tensor:
        """
        Compute the LMI for the discrete-time Lyapunov equation.

        Returns:
            Tensor: The positive definite LMI matrix.
        """
        Q = self.alpha**2 * self.P - self.A.t().matmul(self.P).matmul(self.A)
        # assert isSDP(Q), "The forward result is not positive definite."
        return Q

    @classmethod
    def solve(cls, A: Tensor, alpha: float, tol: float = 1e-9, solver: str = 'MOSEK') -> Tuple[Tensor, Tensor]:
        """
        Solve the discrete-time Lyapunov LMI using cvxpy.

        Args:
            A (Tensor): The system matrix A.
            alpha (float): The decay rate alpha for alpha-stability.
            tol (float): The tolerance for the solver.
            solver (str): The solver to use for cvxpy.

        Returns:
            Tuple[Tensor, Tensor]: The positive definite solution matrix P and the corresponding bounds.
        """
        n = A.size(0)
        A_np = A.numpy()
        P = cp.Variable((n, n), symmetric=True)
        constraints = [P >> tol * np.eye(n), alpha**2 * P - A_np.T @ P @ A_np >> tol * np.eye(n)]
        prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
        prob.solve(solver=solver)

        if prob.status not in ["infeasible", "unbounded"]:
            P_value = torch.tensor(P.value)
            # assert isSDP(P_value), "The solved P is not positive definite."
            return P_value, alpha**2 * P_value - A.t().matmul(P_value).matmul(A)
        else:
            raise ValueError("The LMI is not satisfied for the given system matrix A.")


class LyapunovContinuous(LMI):
    """
    Lyapunov LMI for continuous-time systems.
    """

    def __init__(self, A: Tensor, alpha: float) -> None:
        """
        Initialize the Lyapunov LMI for continuous-time systems.

        Args:
            A (Tensor): The system matrix A.
            alpha (float): The decay rate alpha for alpha-stability.
        """
        super().__init__()
        self.A = Parameter(A)
        self.alpha = alpha
        self.P = Parameter(torch.eye(A.size(0)))

    def forward(self) -> Tensor:
        """
        Compute the LMI for the continuous-time Lyapunov equation.

        Returns:
            Tensor: The positive definite LMI matrix.
        """
        Q = -(self.A.t().matmul(self.P) + self.P.matmul(self.A) + 2 * self.alpha * self.P)
        # assert isSDP(Q), "The forward result is not positive definite."
        return Q

    @classmethod
    def solve(cls, A: Tensor, alpha: float, tol: float = 1e-9, solver: str = 'MOSEK') -> Tuple[Tensor, Tensor]:
        """
        Solve the continuous-time Lyapunov LMI using cvxpy.

        Args:
            A (Tensor): The system matrix A.
            alpha (float): The decay rate alpha for alpha-stability.
            tol (float): The tolerance for the solver.
            solver (str): The solver to use for cvxpy.

        Returns:
            Tuple[Tensor, Tensor]: The positive definite solution matrix P and the corresponding bounds.
        """
        n = A.size(0)
        A_np = A.numpy()
        P = cp.Variable((n, n), symmetric=True)
        constraints = [P >> tol * np.eye(n), A_np.T @ P + P @ A_np + 2 * alpha * P << -tol * np.eye(n)]
        prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
        prob.solve(solver=solver)

        if prob.status not in ["infeasible", "unbounded"]:
            P_value = torch.tensor(P.value)
            # assert isSDP(P_value), "The solved P is not positive definite."
            return -(A.t().matmul(P_value) + P_value.matmul(A) + 2 * alpha * P_value), P_value
        else:
            raise ValueError("The LMI is not satisfied for the given system matrix A.")


# Example usage
if __name__ == "__main__":
    A = torch.tensor([[0.5, 0.1], [0.2, 0.3]])
    alpha = 0.1

    # Discrete-time LMI
    P_discrete, LMI_discrete = LyapunovDiscrete.solve(A, alpha)
    print(f"Discrete-time P: {P_discrete}")
    print(f"Discrete-time LMI: {LMI_discrete}")

    # Continuous-time LMI
    P_continuous, LMI_continuous = LyapunovContinuous.solve(A, alpha)
    print(f"Continuous-time P: {P_continuous}")
    print(f"Continuous-time LMI: {LMI_continuous}")
