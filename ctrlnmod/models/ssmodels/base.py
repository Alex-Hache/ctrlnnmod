from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from geotorch_custom.parametrize import is_parametrized
import torch
from torch import Tensor
from ctrlnmod.utils import FrameCacheManager
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, Any


class SSModel(Module, ABC):
    """
        Abstract base class for state-space models

        Attributes :
            nu (int) : number of inputs
            ny (int) : number of outputs
            nx (int) : number of states
            nd (int, optional) : number of disturbances/exogenous signals
    """

    def __init__(self, nu: int, ny:int, nx:int, nd: Optional[int] = None):
        super(SSModel, self).__init__()
        self.nu = nu
        self.ny = ny
        self.nx = nx

        if nd is not None:
            self.nd = nd

    @abstractmethod
    def forward(self, u, x, d=None):
        pass

    @abstractmethod
    def _frame(self):
        """
            This methods is the junsction from parameter space to weights space
            for non-parameterized models it is the identity operator.
        """
        pass 

    @abstractmethod
    def right_inverse_(self):
        """
            From given weights that belongs to the manifold we initialize the parameter space
        """
        pass

    @abstractmethod
    def init_weights_(self):
        """
            This method enables to initializ the weights it is both a wrapper for irght_inverse and
            the not parameterized weights of the module
        """
        pass