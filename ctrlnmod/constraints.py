from torch.nn import Module
from abc import ABC, abstractmethod
from .lmis import LMI


class _Constraint(ABC, Module):
    '''
        This is an abstract class for constraints
    '''
    # updatable : bool
    # module
    # check method() -> bool
    def __init__(self, module: Module, updatable: bool = False) -> None:
        r'''
            Here intialize at attributes of the class the different tensors needed to
            build the lmi and the potential bound you are enforcing.
        '''
        ABC.__init__(self)
        Module.__init__(self)
        self.module = module
        self.updatable = updatable

    @abstractmethod
    def check(self) -> bool:
        pass

    def is_updatable(self) -> bool:
        return self.updatable


class ManifoldConstraint(_Constraint):
    def __init__(self, module: Module) -> None:
        super(ManifoldConstraint, self).__init__(module, updatable=False)

    def check(self) -> bool:
        return super().check()


class LMIConstraint(_Constraint):
    def __init__(self, module: LMI, updatable: bool = False) -> None:
        super().__init__(module, updatable)

    def check(self):
        return self.module.check_()
