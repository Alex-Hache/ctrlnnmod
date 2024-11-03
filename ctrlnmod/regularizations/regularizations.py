from abc import ABC, abstractmethod
import torch
from torch.linalg import slogdet
import torch.nn as nn
from torch import Tensor
from typing import List, Callable
from ..lmis.base import LMI


class Regularization(ABC):
    def __init__(self, model: nn.Module, factor: float, updatable: bool = True, verbose: bool = False):
        self.model = model
        self.factor = factor
        self.updatable = updatable
        self.verbose = verbose

    @abstractmethod
    def __call__(self) -> Tensor:
        pass

    @abstractmethod
    def update(self) -> None:
        pass


class L1Regularization(Regularization):
    def __init__(self, model: nn.Module, lambda_l1: float, update_factor: float, updatable: bool = True, verbose: bool = False) -> None:
        super().__init__(model, update_factor, updatable, verbose)
        self.lambda_l1 = Tensor([lambda_l1])

    def __call__(self) -> Tensor:
        l1_norm = sum(param.abs().sum() for param in self.model.parameters())
        return self.lambda_l1 * l1_norm

    def update(self) -> None:
        if self.updatable:
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.factor * param.sign()
            if self.verbose:
                print(f"Updated L1 regularization lambda: {self.lambda_l1.item()}")


class L2Regularization(Regularization):
    def __init__(self, model: nn.Module, lambda_l2: float, update_factor: float, updatable: bool = True, verbose: bool = False) -> None:
        super().__init__(model, update_factor, updatable, verbose)
        self.lambda_l2 = Tensor([lambda_l2])

    def __call__(self) -> Tensor:
        l2_norm = sum(param.pow(2).sum() for param in self.model.parameters())
        return self.lambda_l2 * l2_norm

    def update(self) -> None:
        if self.updatable:
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.factor * param
            if self.verbose:
                print(f"Updated L2 regularization lambda: {self.lambda_l2.item()}")

class GammaRegularization(Regularization):
    def __init__(self, model, lambda_gamma, factor, updatable = True, verbose = False):
        super().__init__(model, factor, updatable, verbose)
        self.lambda_gamma = Tensor([lambda_gamma])
        
    
    def __call__(self):
        return self.factor * self.model.gamma ** 2
    
    def update(self) -> None:
        if self.updatable and self.lambda_logdet > self.min_weight:
            old_lambda = self.lambda_logdet.item()
            self.lambda_logdet *= self.factor
            if self.verbose:
                print(f"Updated Gamma regularization lambda: {old_lambda} -> {self.lambda_logdet.item()} \n")
            
class LogdetRegularization(Regularization):
    def __init__(self, lmi: LMI, lambda_logdet: float, update_factor: float, updatable: bool = True, min_weight: float = 1e-6, verbose: bool = False) -> None:
        super().__init__(lmi, update_factor, updatable, verbose)
        self.lambda_logdet = Tensor([lambda_logdet])
        self.min_weight = min_weight

    def __call__(self) -> Tensor:
        matrices = self.model()
        total_logdet = 0
        for matrix in matrices:
            _, logdet = slogdet(matrix)
            total_logdet += logdet
        
        return -self.lambda_logdet * total_logdet

    def update(self) -> None:
        if self.updatable and self.lambda_logdet > self.min_weight:
            old_lambda = self.lambda_logdet.item()
            self.lambda_logdet *= self.factor
            if self.verbose:
                print(f"Updated Logdet regularization lambda: {old_lambda} -> {self.lambda_logdet.item()} \n")
            if self.lambda_logdet <= self.min_weight:
                print("Minimum weight reached")


class StateRegularization(Regularization):
    def __init__(self, model: nn.Module, lambda_state: float, update_factor: float, updatable: bool = True, verbose: bool = False) -> None:
        super().__init__(model, update_factor, updatable, verbose)
        self.lambda_state = Tensor([lambda_state])
        self.criterion = nn.MSELoss()

    def __call__(self, **kwargs) -> Tensor:
        if 'x_pred' not in kwargs or 'x_true' not in kwargs:
            raise ValueError("StateRegularization requires 'x_pred' and 'x_true' arguments")

        x_pred, x_true = kwargs['x_pred'], kwargs['x_true']
        return self.lambda_state * self.criterion(x_pred, x_true)

    def update(self) -> None:
        if self.updatable:
            old_lambda = self.lambda_state.item()
            self.lambda_state *= self.factor
            if self.verbose:
                print(f"Updated State regularization lambda: {old_lambda} -> {self.lambda_state.item()}")


def add_regularization(criterion: Callable[[Tensor, Tensor], Tensor], regularizers: List[Regularization]) -> Callable[[Tensor, Tensor], Tensor]:
    def wrapped_criterion(output: Tensor, target: Tensor) -> Tensor:
        loss = criterion(output, target)
        reg_loss = sum(reg() for reg in regularizers)
        return loss + reg_loss

    return wrapped_criterion
