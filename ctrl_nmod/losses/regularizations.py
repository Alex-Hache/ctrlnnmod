from abc import ABC, abstractmethod
from torch.nn import Module, MSELoss
import torch
from typeguard import typechecked
from ctrl_nmod.lmis.base import LMI
from torch import Tensor


@typechecked
class _Regularization(ABC, Module):
    r"""
        This is base class for all regularizations
    """

    def __init__(self, module: Module, weight: float, scaler: float, updatable: bool) -> None:
        ABC.__init__(self)
        Module.__init__(self)
        self.module = module
        self._weight = weight
        self._scaler = scaler
        self.updatable = updatable

    @abstractmethod
    def forward(self, *args) -> torch.Tensor:
        pass

    def get_weight(self) -> float:
        return self._weight

    def get_scaler(self) -> float:
        return self._scaler


@typechecked
class LMILogdet(_Regularization):
    r"""
        This regularization is a LMI regularisation.
        A semidefinite matrix M > 0 is computed from a specific module attached to it.
        Then the regularization is -logdet(M)
    """

    def __init__(self, M: LMI, weight: float, scale: float = 1.0, updatable: bool = True, min_weight=1e-6) -> None:
        super(LMILogdet, self).__init__(M, weight, scaler=scale, updatable=updatable)
        self.min_weight = min_weight
        self.is_legal = True

    def forward(self) -> torch.Tensor:
        return -torch.logdet(self.module())

    def _update(self) -> None:
        if self.updatable and self._weight > self.min_weight:
            print(f"Updating weight for barrier logdet function : from {self._weight} to {self._weight*self._scaler}")
            self._weight *= self._scaler
            if self._weight <= self.min_weight:
                print("Minimum weight reached")


@typechecked
class StateRegMSE(_Regularization):
    r"""
        This regularization is used to enforce some continuity/consistency
        constraints on the state's initial condition we migh want to learn
        or if state is accessible from data.
    """

    def __init__(self, alpha: float, scale: float = 1.0, updatable=False) -> None:
        loss = MSELoss()
        super().__init__(loss, alpha, scaler=scale, updatable=updatable)

    def forward(self, x_true, x_pred) -> Tensor:
        return self.module(x_true, x_pred)

    def _update(self):
        if self.updatable:
            self._weight *= self._scaler


'''
from abc import ABC, abstractmethod
from torch.nn import Module, ModuleList
import torch
from lmis import lmis
from typeguard import typechecked


@typechecked
class Regularization(ABC, Module):
    r"""
        This is base class for all regularizations
    """

    def __init__(self, module: Module, weight: float) -> None:
        super().__init__()
        self.module = module
        self.__weight = weight
        self._check_module()

    def forward(self):
        return self._compute()*self.__weight

    @abstractmethod
    def _compute(self) -> torch.Tensor:
        pass


    def _check_module(self):
        pass


@typechecked
class LMI_Regularization(Regularization):
    r"""
        This regularization is a LMI regularisation.
        A semidefinite matrix M > 0 is computed from a specific module attached to it
    """

    def _check_module(self):
        if not hasattr(self, 'eval_'):
            raise RuntimeError("Must provide a module with eval func !")

    def _compute(self) -> torch.Tensor:
        return self.module.eval_() # type: ignore
'''
