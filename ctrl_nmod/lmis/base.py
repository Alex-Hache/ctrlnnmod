from abc import ABC, abstractmethod
from typing import Tuple
from torch.nn import Module
from torch import Tensor
from typeguard import typechecked
from linalg.utils import isSDP


@typechecked
class LMI(ABC, Module):
    r"""
        This is base class for all Linear Matrix Inequalities
        A Linear Matrix inequality is built from weights of a Pytorch Module
    """

    def __init__(self) -> None:
        r'''
            Here intialize at attributes of the class the different tensors needed to
            build the lmi and the potential bound you are enforcing.
        '''
        ABC.__init__(self)
        Module.__init__(self)

    @abstractmethod
    def forward(self) -> Tensor:
        r'''
            This method returns M as a positive definite matrix
        '''
        pass

    @abstractmethod
    @classmethod
    def solve(cls, *tensors: Tuple[Tensor], solver: str, tol: float) -> Tuple[Tensor, Tensor]:
        r'''
            This method returns the LMI and the corresponding bounds and certificates
            we aim at evaluate
        '''
        pass

    def check_(self, tol=1e-9) -> bool:
        r'''
            This function given the LMI class return True if the Matrix
            is positive semidefinite with respect to a used-defiend tolerance.
        '''
        return True if isSDP(self.forward(), tol=tol) else False
