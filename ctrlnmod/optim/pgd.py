import torch
from torch.optim import Optimizer
from typing import Optional, List, Union, Callable


def project_to_pos_def(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Projects a symmetric matrix to the closest positive definite matrix.

    This function symmetrizes the input matrix, then performs an eigen-decomposition,
    clips the eigenvalues to a minimum positive threshold (1e-6), and reconstructs
    the matrix to ensure positive definiteness.

    Args:
        matrix (torch.Tensor): A square matrix (assumed symmetric or nearly symmetric).

    Returns:
        torch.Tensor: The closest positive definite matrix.
    """
    matrix_sym = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix_sym)
    eigenvalues = torch.clip(eigenvalues, min=1e-6)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T


class ProjectedOptimizer(Optimizer):
    r"""
    Optimizer wrapper that applies a projection function on specified model parameters
    after each optimization step.

    This optimizer delegates optimization to a wrapped PyTorch optimizer and then
    applies a user-defined projection function (e.g. projection onto positive definite matrices)
    on selected parameters to enforce constraints.

    Args:
        optimizer (torch.optim.Optimizer): The base optimizer to wrap.
        project (Callable[[torch.Tensor], torch.Tensor]): Projection function applied
            to the selected parameters after each step.
        model (torch.nn.Module): The model whose parameters are being optimized.
        modules (Optional[List[Union[str, torch.nn.Module]]], optional): List of parameter
            names or submodules to which projection is applied. If None, no projection
            is applied. Defaults to None.

    Example:
        >>> import torch.nn as nn
        >>> import torch.optim as optim
        >>>
        >>> model = MyModel()
        >>> base_optimizer = optim.Adam(model.parameters(), lr=0.01)
        >>>
        >>> # Project parameters with names 'layer1.weight' and 'layer2.weight' to positive definite matrices
        >>> proj_opt = ProjectedOptimizer(
        ...     base_optimizer,
        ...     project=project_to_pos_def,
        ...     model=model,
        ...     modules=['layer1.weight', 'layer2.weight']
        ... )
        >>>
        >>> def closure():
        ...     proj_opt.zero_grad()
        ...     output = model(input)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        ...     return loss
        >>>
        >>> for input, target in data_loader:
        ...     proj_opt.step(closure)
    """

    def __init__(self, optimizer: Optimizer, project: Callable[[torch.Tensor], torch.Tensor], model: torch.nn.Module,
                 modules: Optional[List[Union[str, torch.nn.Module]]] = None):
        self.optimizer = optimizer
        self.project = project
        self.model = model
        self.modules = set()

        if modules is not None:
            for module in modules:
                if isinstance(module, str):
                    self.modules.add(module)
                elif isinstance(module, torch.nn.Module):
                    # Add all parameter names of the submodule
                    for name, _ in module.named_parameters():
                        self.modules.add(name)
                else:
                    raise ValueError(f"Unsupported type in modules: {type(module)}")

    def step(self, closure=None):
        """
        Performs a single optimization step, then applies the projection function
        to the selected parameters.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            The loss returned by the closure, if any.
        """
        loss = self.optimizer.step(closure)

        if self.project is not None:
            for name, param in self.model.named_parameters():
                if name in self.modules:
                    param.data = self.project(param.data)

        return loss

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, value):
        self.optimizer.state = value
