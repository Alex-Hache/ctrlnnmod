from torch.nn import Module
from torch import Tensor
from ctrlnmod.utils import FrameCacheManager
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class SSModel(Module, ABC):
    """
        Abstract base class for state-space models.

        Attributes :
            nu (int) : number of inputs
            ny (int) : number of outputs
            nx (int) : number of states
            nd (int, optional) : number of disturbances/exogenous signals
    """

    def __init__(self, nu: int, ny:int, nx:int, nd: Optional[int] = None):
        """
            Args:
                nu (int): Number of inputs.
                ny (int): Number of outputs.
                nx (int): Number of states.
                nd (int, optional): Number of disturbances/exogenous signals. Defaults to None.
        """
        if not isinstance(nu, int) or nu < 0:
            raise ValueError("nu must be a non-negative integer")
        if not isinstance(ny, int) or ny < 0:
            raise ValueError("ny must be a non-negative integer")
        if not isinstance(nx, int) or nx < 0:
            raise ValueError("nx must be a non-negative integer")
        
        super(SSModel, self).__init__()
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nd = nd

        self._frame_cache = FrameCacheManager()

    @abstractmethod
    def forward(self, u, x, d=None):
        """
            Forward pass of the model. This method should be implemented by subclasses.
            Args:
                u (Tensor): Input tensor.
                x (Tensor): State tensor.
                d (Tensor, optional): Disturbance tensor. Defaults to None.
    
            Returns:
                Tensor: Output tensor.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def __repr__(self):
        """
            String representation of the model.
        """
        if hasattr(self, 'nd'):
            return f"{self.__class__.__name__}(nu={self.nu}, ny={self.ny}, nx={self.nx}, nd={self.nd})"
        else:
            return f"{self.__class__.__name__}(nu={self.nu}, ny={self.ny}, nx={self.nx})"
    
    def __str__(self):
        """
            String representation of the model.
        """
        return self.__repr__()

    @abstractmethod
    def _frame(self) -> Tuple[Tensor, ...]:
        """
            This methods is the junsction from parameter space to weights space
            for non-parameterized models it is the identity operator.
        """
        pass 

    @abstractmethod
    def _right_inverse(self, *args, **kwargs):
        """
            From given weights that belongs to the manifold we initialize the parameter space
        """
        pass

    @abstractmethod
    def init_weights_(self, *args, **kwargs):
        """
            This method enables to initialize the weights it is both a wrapper for irght_inverse and
            the not parameterized weights of the module
        """
        pass

    @abstractmethod
    def clone(self):
        """
            Clone the model, it has to be implemented to be compliant with simulator classes.
        """
        raise NotImplementedError("State-space models must implement a clone method")