from torch.nn import Module


def find_module(model: Module, target_class):
    for name, module in model.named_children():
        if isinstance(module, target_class):
            # print(f"Le module '{name}' est une instance de la classe '{target_class.__name__}'.")
            return module
        if list(module.children()):
            found_module = find_module(module, target_class)
            if found_module is not None:
                return found_module
    return None
