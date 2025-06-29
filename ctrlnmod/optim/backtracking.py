import torch.optim as optim
import torch.nn as nn
from typing import Callable


class BackTrackOptimizer:
    """
    Wrapper optimizer implementing backtracking line search to enforce a condition on updated parameters.

    This optimizer wraps a PyTorch optimizer and performs a backtracking line search after each step.
    After performing the step, it checks a user-provided condition function on the updated model parameters.
    If the condition is not satisfied, it rolls back the parameters and reduces the learning rate by
    multiplying it by `beta`, then retries the step. This process repeats until the condition is met or
    the maximum number of backtracking iterations is reached.

    Args:
        optimizer (torch.optim.Optimizer): The base optimizer to wrap.
        module (nn.Module): The model whose parameters are being optimized.
        condition_fn (Callable[[nn.Module], bool]): A callable that receives the model and returns
            True if the updated parameters satisfy the acceptance condition.
        beta (float, optional): Multiplicative factor to decrease learning rate during backtracking (default: 0.5).
        max_iter (int, optional): Maximum number of backtracking iterations (default: 20).

    Attributes:
        n_backtrack_iter (int): The number of backtracking iterations performed.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.optim as optim
        >>>
        >>> model = MyModel()
        >>> criterion = nn.MSELoss()
        >>> base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        >>>
        >>> def condition_fn(mod):
        ...     # Example condition: loss decreases after step
        ...     output = mod(input)
        ...     loss = criterion(output, target)
        ...     return loss.item() < old_loss
        >>>
        >>> optimizer = BackTrackOptimizer(base_optimizer, model, condition_fn)
        >>>
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        ...     return loss
        >>>
        >>> for input, target in data_loader:
        ...     old_loss = float('inf')
        ...     optimizer.step(closure)
    """

    def __init__(self, optimizer: optim.Optimizer, module: nn.Module, condition_fn: Callable[[nn.Module], bool], beta=0.5, max_iter=20):
        self.optimizer = optimizer
        self.module = module
        self.condition_fn = condition_fn
        self.beta = beta
        self.max_iter = max_iter
        self.n_backtrack_iter = 0

    def step(self, closure):
        """
        Performs a single optimization step with backtracking line search.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss returned by the closure after the accepted step.
        """
        # Save current parameters state
        state_dict_copy = {k: v.clone() for k, v in self.module.state_dict().items()}

        # Perform initial optimizer step
        loss = self.optimizer.step(closure)

        iter_count = 0

        # Backtracking loop: check condition and retry with smaller lr if necessary
        while not self.condition_fn(self.module) and iter_count < self.max_iter:
            # Roll back parameters
            self.module.load_state_dict(state_dict_copy)

            self.n_backtrack_iter += 1

            # Reduce learning rate by factor beta in all param groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.beta

            # Retry step with reduced learning rate
            loss = self.optimizer.step(closure)

            iter_count += 1

        return loss

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
