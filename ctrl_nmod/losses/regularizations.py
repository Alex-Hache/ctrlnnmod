from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from torch.nn import Module, ModuleList, MSELoss
import torch
from typeguard import typechecked
from ctrl_nmod.lmis.base import LMI
from torch import Tensor


@typechecked
class _Regularization(ABC, Module):
    r"""
        This is base class for all regularizations
    """

    def __init__(self, module: Module, weight: float, scaler: float) -> None:
        ABC.__init__(self)
        Module.__init__(self)
        self.module = module
        self._weight = weight
        self._scaler = scaler

    def forward(self):
        return self._compute() * self._weight

    @abstractmethod
    def _compute(self) -> torch.Tensor:
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

    def __init__(self, M: LMI, weight: float, scale: float = 1.0) -> None:
        super(LMILogdet, self).__init__(M, weight, scaler=scale)

    def _compute(self) -> torch.Tensor:
        return -torch.logdet(self.module())

    def _update(self) -> None:
        self._weight *= self._scaler


@typechecked
class StateRegMSE(_Regularization):
    r"""
        This regularization is used to enforce some continuity/consistency
        constraints on the state's initial condition we migh want to learn
        or if state is accessible from data.
    """

    def __init__(self, alpha: float, scale: float = 1.0) -> None:
        loss = MSELoss()
        super().__init__(loss, alpha, scaler=scale)

    def _compute(self, x_true, x_pred) -> Tensor:
        return self.module(x_true, x_pred)

    def _update(self):
        self._weight *= self._scaler


@typechecked
class RegularizationsList(ModuleList):

    _regularizations: List[_Regularization]

    def __init__(self, regularizations: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if regularizations is not None:
            self += regularizations

    def update(self):
        for reg in self:
            if hasattr(reg, '_update'):
                reg._update()


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
