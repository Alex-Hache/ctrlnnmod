from abc import ABC, abstractmethod
from typing import Tuple, Callable
from torch.nn import Module
from torch import Tensor
from typeguard import typechecked
from ..linalg.utils import is_positive_definite


@typechecked
class LMI(ABC, Module):
    """
    Base class for all Linear Matrix Inequalities (LMI).
    An LMI is built from the weights of a PyTorch Module.
    Subclasses must implement the specific attributes/submatrices they need to compute the forward method.
    """

    def __init__(self, module: Module, extract_matrices: Callable) -> None:
        ABC.__init__(self)
        Module.__init__(self)

        # For now the module itself is not used but it makes the point clearer that an LMI
        # comes from the weights of a specific module and it is useful for debugging gradient problems
        self.module = module
        self.extract_matrices = extract_matrices
        self.hook = self.register_forward_pre_hook(self._update_matrices) # type: ignore

    @abstractmethod
    def forward(self) -> Tuple[Tensor, ...]:
        """
        Returns M as a positive definite matrix.

        Returns:
            Tensor: A positive definite matrix.
        """
        pass

    @classmethod
    @abstractmethod
    def solve(cls, *args, **kwargs) -> Tuple[Tensor, ...]:
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

    @abstractmethod
    def _update_matrices(self) -> None:
        pass


    def check_(self, tol: float = 1e-9) -> bool:
        """
        Checks if the matrix is positive semidefinite within a user-defined tolerance.

        Args:
            tol (float): The tolerance for checking positive semidefiniteness.

        Returns:
            bool: True if the matrix is positive semidefinite within the given tolerance, False otherwise.
        """

        matrices = self.forward()
        if not isinstance(matrices, tuple):
            matrices = (matrices,)
        return all(is_positive_definite(matrix) for matrix in matrices)
