from abc import ABC, abstractmethod
import torch
from torch.linalg import slogdet
import torch.nn as nn
from torch import Tensor
from typing import List, Callable
from ..lmis.base import LMI


class Regularization(ABC, nn.Module):
    def __init__(self, model: nn.Module, factor: float, updatable: bool = True, verbose: bool = False):
        ABC.__init__(self)
        nn.Module.__init__(self)
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

class DDRegularization(Regularization):
    def __init__(self, lmi: LMI, lambda_dd: float, update_factor: float, actf='lse', updatable = True, verbose = False, e= 0.1):
        super().__init__(lmi, update_factor, updatable, verbose)
        self.lambda_dd = lambda_dd
        if actf == 'relu':
            self.actf = nn.ReLU()
        elif actf == 'lse':
            self.actf = torch.logsumexp
        else:
            raise ValueError(f"Only ReLU or Logsum exp functions are accepted found : {actf}")

        # Creation of the Z matrix for the constraints
        matrices = self.model() 
        self.Z_matrices = nn.ParameterList([
            nn.Parameter(torch.eye(mat.shape[0], mat.shape[0]).requires_grad_(True))
            for mat in matrices
        ])
        self.e = e


    def __call__(self):
        matrices = self.model()  # Returns a tuple of matrices
        
        all_cons1 = []
        all_cons2 = []
        all_cons3 = []
        
        # Process each matrix from the tuple
        for mat, Z in zip(matrices, self.Z_matrices):
            # Compute diagonal matrix
            mat_diag = torch.diag(torch.diag(mat))
            
            # Compute W matrix (full - diagonal)
            W = mat - mat_diag
            
            # Constraint 1: -Z - W ≤ 0
            cons1 = (-Z - W).flatten()
            all_cons1.append(cons1)
            
            # Constraint 2: W - Z ≤ 0
            cons2 = (W - Z).flatten()
            all_cons2.append(cons2)
            
            # Constraint 3: sum(Z, dim=1) = diag(M)
            cons3 = torch.sum(Z, dim=1) - torch.diag(mat)
            all_cons3.append(cons3)
        
        # Concatenate all constraints
        Cons1 = torch.cat(all_cons1)
        Cons2 = torch.cat(all_cons2)
        Cons3 = torch.cat(all_cons3)

        
        # Concatenate all constraints
        constraints = torch.cat([Cons1, Cons2, Cons3])
        
        # Apply the selected activation function
        if isinstance(self.actf, nn.ReLU):
            # For ReLU, simply sum the activated constraints
            return torch.sum(self.actf(constraints))
        else:
            # For logsumexp
            return self.lambda_dd * self.e * self.actf(constraints / self.e, dim=0)
    
    def update(self):
        if self.updatable:
            self.lambda_dd = self.factor * self.lambda_dd
            if self.verbose:
                print(f"Updated DD regularization lambda: {self.lambda_dd}")

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
