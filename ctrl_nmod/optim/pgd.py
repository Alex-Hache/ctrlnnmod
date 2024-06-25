import torch
from torch.optim import Optimizer


def project_to_pos_def(matrix):
    """ Projects a symmetric matrix to the closest positive definite matrix """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # Clip eigenvalues to a small positive number
    eigenvalues = torch.clip(eigenvalues, min=1e-6, max=None)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T


class ProjectedOptimizer(Optimizer):
    def __init__(self, optimizer, project=None):
        self.optimizer = optimizer
        self.project = project

    def step(self, closure=None):
        # Perform a single optimization step
        loss = self.optimizer.step(closure)

        # Apply the projection to each parameter if provided
        if self.project is not None:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data = self.project(p.data)

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
