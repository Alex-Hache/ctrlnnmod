from abc import ABC, abstractmethod
from typing import Tuple
from torch.nn import Module
from torch import Tensor
from typeguard import typechecked
from ..linalg.utils import isSDP


@typechecked
class LMI(ABC, Module):
    """
    Base class for all Linear Matrix Inequalities (LMI).
    An LMI is built from the weights of a PyTorch Module.
    """

    def __init__(self) -> None:
        """
        Initialize the attributes of the class.
        This includes the tensors needed to build the LMI and the potential bound being enforced.
        """
        ABC.__init__(self)
        Module.__init__(self)

    @abstractmethod
    def forward(self) -> Tensor:
        """
        Returns M as a positive definite matrix.

        Returns:
            Tensor: A positive definite matrix.
        """
        pass

    @classmethod
    @abstractmethod
    def solve(cls, *tensors: Tuple[Tensor], solver: str, tol: float) -> Tuple[Tensor, Tensor]:
        """
        Returns the LMI and the corresponding bounds and certificates for evaluation.

        Args:
            tensors (Tuple[Tensor]): A tuple of tensors required for solving the LMI.
            solver (str): The solver to be used for LMI.
            tol (float): The tolerance for the solver.

        Returns:
            Tuple[Tensor, Tensor]: The solution of the LMI and the corresponding bounds.
        """
        pass

    def check_(self, tol: float = 1e-9) -> bool:
        """
        Checks if the matrix is positive semidefinite within a user-defined tolerance.

        Args:
            tol (float): The tolerance for checking positive semidefiniteness.

        Returns:
            bool: True if the matrix is positive semidefinite within the given tolerance, False otherwise.
        """
        return isSDP(self.forward(), tol=tol)
