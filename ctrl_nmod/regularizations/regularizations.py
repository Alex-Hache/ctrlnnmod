from abc import ABC, abstractmethod
import torch
from torch.linalg import slogdet
import torch.nn as nn
from torch import Tensor
from typing import List, Callable
from ..lmis.base import LMI


class Regularization(ABC):
    """
    Abstract base class for regularizations.

    Attributes:
        model (torch.nn.Module): The model to which regularization is applied.
        factor (float): The factor by which weights are adjusted during update.
        updatable (bool): Flag indicating if the regularization factor is updatable.
    """

    def __init__(self, model: nn.Module, factor: float, updatable: bool = True):
        """
        Initializes the Regularization class.

        Args:
            model (torch.nn.Module): The model to which regularization is applied.
            factor (float): The factor by which weights are adjusted during update.
            updatable (bool): Flag indicating if the regularization factor is updatable.
        """
        self.model = model
        self.factor = factor
        self.updatable = updatable

    @abstractmethod
    def __call__(self) -> Tensor:
        """
        Computes the regularization term.

        Returns:
            torch.Tensor: The computed regularization term.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Updates the regularization factor if updatable.

        """
        pass


class L1Regularization(Regularization):
    """
    L1 regularization class.

    Attributes:
        model (torch.nn.Module): The model to which regularization is applied.
        lambda_l1 (float): The L1 regularization coefficient.
        factor (float): The factor by which weights are adjusted during update.
        updatable (bool): Flag indicating if the regularization factor is updatable.
    """

    def __init__(self, model: nn.Module, lambda_l1: float, update_factor: float, updatable: bool = True) -> None:
        """
        Initializes the L1Regularization class.

        Args:
            model (torch.nn.Module): The model to which regularization is applied.
            lambda_l1 (float): The L1 regularization coefficient.
            update_factor (float): The factor by which weights are adjusted during update.
            updatable (bool): Flag indicating if the regularization factor is updatable.
        """
        super().__init__(model, update_factor, updatable)
        self.lambda_l1 = Tensor([lambda_l1])

    def __call__(self) -> Tensor:
        """
        Computes the L1 regularization term.

        Returns:
            torch.Tensor: The computed L1 regularization term.
        """
        l1_norm = sum(param.abs().sum() for param in self.model.parameters())
        return self.lambda_l1 * l1_norm

    def update(self) -> None:
        """
        Updates the regularization weights by subtracting the factor times the sign of the parameters.

        """
        if self.updatable:
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.factor * param.sign()


class L2Regularization(Regularization):
    """
    L2 regularization class.

    Attributes:
        model (torch.nn.Module): The model to which regularization is applied.
        lambda_l2 (float): The L2 regularization coefficient.
        factor (float): The factor by which weights are adjusted during update.
        updatable (bool): Flag indicating if the regularization factor is updatable.
    """

    def __init__(self, model: nn.Module, lambda_l2: float, update_factor: float, updatable: bool = True) -> None:
        """
        Initializes the L2Regularization class.

        Args:
            model (torch.nn.Module): The model to which regularization is applied.
            lambda_l2 (float): The L2 regularization coefficient.
            update_factor (float): The factor by which weights are adjusted during update.
            updatable (bool): Flag indicating if the regularization factor is updatable.
        """
        super().__init__(model, update_factor, updatable)
        self.lambda_l2 = Tensor([lambda_l2])

    def __call__(self) -> Tensor:
        """
        Computes the L2 regularization term.

        Returns:
            torch.Tensor: The computed L2 regularization term.
        """
        l2_norm = sum(param.pow(2).sum() for param in self.model.parameters())
        return self.lambda_l2 * l2_norm

    def update(self) -> None:
        """
        Updates the regularization weights by subtracting the factor times the parameters.

        """
        if self.updatable:
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.factor * param


class LogdetRegularization(Regularization):
    """
    Log-determinant regularization class.

    Attributes:
        model (torch.nn.Module): The model to which regularization is applied.
        lambda_logdet (float): The log-determinant regularization coefficient.
        factor (float): The factor by which weights are adjusted during update.
        updatable (bool): Flag indicating if the regularization factor is updatable.
        min_weight (float): Minimum weight for regularization.
    """

    def __init__(self, lmi: LMI, lambda_logdet: float, update_factor: float, updatable: bool = True, min_weight: float = 1e-6) -> None:
        """
        Initializes the LogdetRegularization class.

        Args:
            lmi (torch.nn.Module): The lmi to which regularization is applied.
            lambda_logdet (float): The log-determinant regularization coefficient.
            factor (float): The factor by which weights are adjusted during update.
            updatable (bool): Flag indicating if the regularization factor is updatable.
            min_weight (float): Minimum weight for regularization.
        """
        super().__init__(lmi, update_factor, updatable)
        self.lambda_logdet = Tensor([lambda_logdet])
        self.min_weight = min_weight

    def __call__(self) -> Tensor:
        """
        Computes the log-determinant regularization term.

        Returns:
            torch.Tensor: The computed log-determinant regularization term.

        Raises:
            ValueError: If the matrix is not positive definite.
        """
        matrix = self.model()
        sign, logdet = slogdet(matrix)
        if sign <= 0:
            raise ValueError("Matrix is not positive definite.")
        return -self.lambda_logdet * logdet

    def update(self) -> None:
        """
        Updates the regularization factor if it is updatable and above the minimum weight.

        """
        if self.updatable and self.factor > self.min_weight:
            print("Updating weight parameter")
            self.lambda_logdet *= self.factor
            if self.factor <= self.min_weight:
                print("Minimum weight reached")


class StateRegularization(Regularization):
    """
    State regularization class penalizing the loss function according to an MSE between predicted x and given x.

    Attributes:
        model (nn.Module): The model to which regularization is applied.
        lambda_state (float): The state regularization coefficient.
        factor (float): The factor by which weights are adjusted during update.
        updatable (bool): Flag indicating if the regularization factor is updatable.
    """

    def __init__(self, model: nn.Module, lambda_state: float, update_factor: float, updatable: bool = True) -> None:
        """
        Initializes the StateRegularization class.

        Args:
            model (nn.Module): The model to which regularization is applied.
            lambda_state (float): The state regularization coefficient.
            update_factor (float): The factor by which weights are adjusted during update.
            updatable (bool): Flag indicating if the regularization factor is updatable.
        """
        super().__init__(model, update_factor, updatable)
        self.lambda_state = Tensor([lambda_state])
        self.criterion = nn.MSELoss()

    def __call__(self, x_pred: Tensor, x_true: Tensor) -> Tensor:
        """
        Computes the state regularization term.

        Args:
            x_pred (Tensor): The predicted state values.
            x_true (Tensor): The true state values.

        Returns:
            Tensor: The computed state regularization term.
        """
        return self.lambda_state * self.criterion(x_pred, x_true)

    def update(self) -> None:
        """
        Updates the regularization weights if updatable.
        """
        if self.updatable:
            self.lambda_state *= self.factor


def add_regularization(criterion: Callable[[Tensor, Tensor], Tensor], regularizers: List[Regularization]) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Decorator to add regularization terms to the loss function.

    Args:
        criterion (Callable[[Tensor, Tensor], Tensor]): The base loss function (e.g., nn.MSELoss).
        regularizers (List[Regularization]): A list of regularization terms to be added to the loss.

    Returns:
        Callable[[Tensor, Tensor], Tensor]: A new loss function that includes the regularization terms.
    """
    def wrapped_criterion(output: Tensor, target: Tensor) -> Tensor:
        loss = criterion(output, target)
        reg_loss = sum(reg() for reg in regularizers)
        return loss + reg_loss

    return wrapped_criterion
