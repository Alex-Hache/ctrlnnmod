import torch.optim as optim
import torch.nn as nn
from typing import Callable
import torch
from ctrl_nmod.lmis import LMI


def is_positive_definite(lmi: LMI):
    try:
        _ = torch.linalg.cholesky(lmi())
        return True
    except RuntimeError:
        return False


class BackTrackOptimizer:
    def __init__(self, optimizer: optim.Optimizer, module: nn.Module, condition_fn: Callable, beta=0.8, max_iter=20):
        """
        Initializes the BackTrackOptimizer.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to wrap.
            module (nn.Module): A function that takes current and candidate parameter states
                                     and returns True if the candidate should be accepted.
            beta (float): The beta parameter for backtracking (default: 0.8).
            max_iter (int): The maximum number of backtracking iterations (default: 10).
        """
        self.optimizer = optimizer
        self.module = module
        self.condition_fn = condition_fn
        self.beta = beta
        self.max_iter = max_iter

    def step(self, closure):
        """
        Performs a single optimization step with backtracking line search.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """

        # Save the current state of parameters
        state_dict = self.optimizer.state_dict()

        # Perform a single optimization step with the original optimizer
        self.optimizer.step(closure)

        t = 1.0
        iter_count = 0

        while not self.condition_fn(self.module()) and iter_count < self.max_iter:
            # Restore the previous state
            self.optimizer.load_state_dict(state_dict)

            # Apply the backtracking step
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None and param in self.module():
                        param.data.add_(t * param.grad, alpha=-self.beta)

            t *= self.beta
            iter_count += 1

        # If max iterations reached, revert to the last best state
        if iter_count == self.max_iter:
            self.optimizer.load_state_dict(state_dict)
        else:
            # Update the optimizer's state to reflect the new parameters
            self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()

# Example usage:
# model = ...
# criterion = ...
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# condition_fn = lambda old_loss, new_loss: new_loss < old_loss
# optimizer = BackTrackOptimizer(optimizer, condition_fn)

# def closure():
#     optimizer.zero_grad()
#     output = model(input)
#     loss = criterion(output, target)
#     loss.backward()
#     return loss

# for input, target in data_loader:
#     optimizer.step(closure)
