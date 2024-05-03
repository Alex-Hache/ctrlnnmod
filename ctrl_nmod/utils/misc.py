import torch


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
