from abc import ABC, abstractmethod
from typing import List
from torch.nn import Module, ModuleList
import torch
from typeguard import typechecked
from ctrl_nmod.lmis.base import LMI


@typechecked
class Regularization(ABC, Module):
    r"""
        This is base class for all regularizations
    """

    def __init__(self, module: Module, weight: float) -> None:
        super(ABC, self).__init__()
        self.module = module
        self.__weight = weight

    def forward(self):
        return self._compute()*self.__weight

    @abstractmethod
    def _compute(self) -> torch.Tensor:
        pass


@typechecked
class LMILogdet(Regularization):
    r"""
        This regularization is a LMI regularisation.
        A semidefinite matrix M > 0 is computed from a specific module attached to it.
        Then the regularization is -logdet(M)
    """

    def __init__(self, lmi: LMI, weight: float) -> None:
        super(LMILogdet, self).__init__(lmi, weight)

    def _compute(self) -> torch.Tensor:
        return -torch.logdet(self.module.forward())


@typechecked
class RegularizationsList(Module):

    def __init__(self, regularizations: List[Regularization]) -> None:
        super(RegularizationsList, self).__init__()
        self.regularizations = ModuleList(regularizations)

    def forward(self):
        reg_loss = torch.zeros((1))
        for regularization in self.regularizations:
            reg_loss += regularization.forward()
        return reg_loss

    def append(self, reg: Regularization):
        self.regularizations.append(reg)

    def remove(self, idx):
        ''''
            Remove the regularization located to the index in the index idx
        '''
        self.regularizations.pop(idx)  # type: ignore


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