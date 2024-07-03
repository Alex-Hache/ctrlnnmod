import torch
from torch.optim import Optimizer
from typing import Optional, List, Union, Callable


def project_to_pos_def(matrix):
    """ Projects a symmetric matrix to the closest positive definite matrix """
    # Step 1: Make the matrix symmetric
    matrix_sym = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix_sym)
    # Clip eigenvalues to a small positive number
    eigenvalues = torch.clip(eigenvalues, min=1e-6, max=None)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T


class ProjectedOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, project: Callable, model: torch.nn.Module,
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
                    for name, _ in module.named_parameters():
                        self.modules.add(name)
                else:
                    raise ValueError(f"Unsupported type in modules: {type(module)}")

    def step(self, closure=None):
        # Perform a single optimization step
        loss = self.optimizer.step(closure)

        # Apply the projection to each specified parameter
        if self.project is not None:
            for name, param in self.model.named_parameters():
                if name in self.modules:
                    param.data = self.project(param.data)

        return loss

    def zero_grad(self):
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
