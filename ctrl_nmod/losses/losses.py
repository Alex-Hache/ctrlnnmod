import torch
from torch import Tensor
from typing import Sequence, Optional
from abc import ABC, abstractmethod
from ..regularizations import Regularization


class BaseLoss(ABC):
    def __init__(self, regularizers: Optional[Sequence[Regularization]] = None):
        """
        Initializes the BaseLoss class with optional regularization terms.

        Args:
            regularizers (List[Regularization], optional): A list of regularization terms to be added to the loss. Defaults to None.
        """
        self.regularizers = regularizers if regularizers else []

    def add_regularization(self, loss: Tensor) -> Tensor:
        """
        Adds regularization terms to the given loss.

        Args:
            loss (Tensor): The base loss.

        Returns:
            Tensor: The loss with regularization terms added.
        """
        if self.regularizers:
            reg_loss = sum(reg() for reg in self.regularizers)
            return loss + reg_loss
        return loss

    @abstractmethod
    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the loss with optional regularization terms.

        Args:
            output (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The computed loss with regularization terms added.
        """
        pass


class MSELoss(BaseLoss):
    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the MSE loss with optional regularization terms.

        Args:
            output (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The computed MSE loss with regularization terms added.
        """
        loss = torch.mean((output - target) ** 2)
        return self.add_regularization(loss)


class NMSELoss(BaseLoss):
    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the NMSE loss with optional regularization terms.

        Args:
            output (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The computed NMSE loss with regularization terms added.
        """
        loss = torch.mean((output - target) ** 2) / torch.mean(target ** 2)
        return self.add_regularization(loss)


class FitPercentLoss(BaseLoss):
    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the fit percentage loss with optional regularization terms.

        Args:
            output (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The computed fit percentage loss with regularization terms added.
        """
        loss = 1 - torch.norm(output - target) / torch.norm(target)
        return self.add_regularization(loss)


class RMSELoss(BaseLoss):
    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the RMSE loss with optional regularization terms.

        Args:
            output (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The computed RMSE loss with regularization terms added.
        """
        loss = torch.sqrt(torch.mean((output - target) ** 2))
        return self.add_regularization(loss)


class NRMSELoss(BaseLoss):
    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the NRMSE loss with optional regularization terms.

        Args:
            output (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The computed NRMSE loss with regularization terms added.
        """
        loss = torch.sqrt(torch.mean((output - target) ** 2)
                          ) / torch.std(target)
        return self.add_regularization(loss)

# Example usage:
# model = YourModel()  # Replace with your model
# regularizers = [L1Regularization(model, lambda_l1=0.01, update_factor=0.1)]
# mse_loss = MSELoss(regularizers)
# loss = mse_loss(output, target)
