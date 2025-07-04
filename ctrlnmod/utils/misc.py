import torch
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Tuple, Union
from torch import Tensor

def parse_act_f(act_f: Union[str, torch.nn.Module]) -> torch.nn.Module:
    """
    Parse the activation function from a string or a torch module.
    
    Args:
        act_f (Union[str, torch.nn.Module]): The activation function as a string or a torch module.
        
    Returns:
        torch.nn.Module: The corresponding activation function module.
    """
    if isinstance(act_f, str):
        if act_f == 'relu':
            return torch.nn.ReLU()
        elif act_f == 'tanh':
            return torch.nn.Tanh()
        elif act_f == 'sigmoid':
            return torch.nn.Sigmoid()
        elif act_f == 'leaky_relu':
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Activation function not implemented yet : {act_f}")
    elif isinstance(act_f, torch.nn.Module):
        return act_f
    else:
        raise TypeError("act_f must be a string or a torch module")

class FrameCacheManager:
    def __init__(self):
        self.cache: Optional[Tuple[Tensor, ...]] = None
        self.is_caching = False
        self._children = []
    
    def register_child(self, child_cache: 'FrameCacheManager'):
        self._children.append(child_cache)

    @contextmanager
    def cache_frame(self):
        """Context manager to store parameterization results"""
        previous_cache_state = self.is_caching
        if not previous_cache_state:
            self.is_caching = True
            for child in self._children:
                child.is_caching = True
        try:
            yield
        finally:
            # Restaurer l'état précédent et nettoyer le cache seulement si on était pas déjà en train de cacher
            if not previous_cache_state:
                self.is_caching = False
                self.cache = None
                for child in self._children:
                    child.is_caching = False
                    child.cache = None

def rk4_discretize(A, h):
    I = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
    k1 = h * A
    k2 = h * torch.mm(A, (I + 0.5 * k1))
    k3 = h * torch.mm(A, (I + 0.5 * k2))
    k4 = h * torch.mm(A, (I + k3))
    return I + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def find_module(model: torch.nn.Module, target_class):
    for name, module in model.named_children():
        if isinstance(module, target_class):
            # print(f"Le module '{name}' est une instance de la classe '{target_class.__name__}'.")
            return module
        if list(module.children()):
            found_module = find_module(module, target_class)
            if found_module is not None:
                return found_module
    return None


def is_legal(v) -> bool:
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal


def write_flat_params(module: torch.nn.Module, x: torch.Tensor):
    r""" Writes vector x to model parameters.."""
    index = 0
    theta = torch.Tensor(x)
    for name, p in module.named_parameters():
        p.data = theta[index:index + p.numel()].view_as(p.data)
        index = index + p.numel()


def flatten_params(module: torch.nn.Module) -> torch.Tensor:
    views = []
    for i, p in enumerate(module.parameters()):
        if p is None:
            view = p.new(p.numel()).zero_()
        elif p.is_sparse:
            view = p.to_dense().view(-1)
        else:
            view = p.view(-1)
        views.append(view)
    return torch.cat(views, 0)


# Définir le décorateur pour le suivi
def monitor_training(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        log = {'loss': [], 'iterations': 0, 'gradients': []}

        def monitor_step(model, loss, optimizer):
            log['loss'].append(loss.item())
            log['iterations'] += 1
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.norm().item())
            log['gradients'].append(gradients)

        result = func(*args, **kwargs, monitor_step=monitor_step)
        result['log'] = log
        return result
    return wrapper


def add_info_decorator(info_key, info_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if 'log' in result:
                result['log'][info_key] = info_value
            return result
        return wrapper
    return decorator


def backtrack(module, criterion, step_ratio=0.5, max_iter=100):
    with torch.no_grad():
        theta0 = flatten_params(module)
        i = 0
        while not is_legal(criterion()) and i <= max_iter:
            theta = theta0 * step_ratio + theta0 * (1 - step_ratio)
            write_flat_params(module, theta)
            i += 1
            if i > max_iter:
                print("Maximum iterations reached")
                criterion.updatable = False
                break

