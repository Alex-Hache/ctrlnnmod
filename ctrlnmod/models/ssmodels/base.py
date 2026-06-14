from torch import Tensor
from abc import abstractmethod
from typing import Dict, Tuple, Optional
from ctrlnmod.blocks.base import Block, get_fqn, import_class


class SSModel(Block):
    """
        Abstract base class for state-space models.

        A state-space model is a *dynamic* :class:`~ctrlnmod.blocks.base.Block`
        whose default ports are ``u`` (and optionally ``d``) as inputs and ``y`` as
        output. This lets every existing model be wired into a
        :class:`~ctrlnmod.diagram.diagram.Diagram` while keeping the historical
        ``forward(u, x, d) -> (dx, y)`` API unchanged.

        Attributes :
            nu (int) : number of inputs
            ny (int) : number of outputs
            nx (int) : number of states
            nd (int, optional) : number of disturbances/exogenous signals
    """

    def __init__(self, nu: int, ny:int, nx:int, nd: Optional[int] = None,
                 feedthrough: bool = False):
        """
            Args:
                nu (int): Number of inputs.
                ny (int): Number of outputs.
                nx (int): Number of states.
                nd (int, optional): Number of disturbances/exogenous signals. Defaults to None.
                feedthrough (bool): Whether ``y`` depends directly on ``u``/``d``.
                    Defaults to ``False`` (strictly-proper ``y = g(x)``), which lets
                    the diagram engine break algebraic loops.
        """
        if not isinstance(nu, int) or nu < 0:
            raise ValueError("nu must be a non-negative integer")
        if not isinstance(ny, int) or ny < 0:
            raise ValueError("ny must be a non-negative integer")
        if not isinstance(nx, int) or nx < 0:
            raise ValueError("nx must be a non-negative integer")

        in_ports = {"u": nu}
        if nd is not None:
            in_ports["d"] = nd
        super().__init__(in_ports=in_ports, out_ports={"y": ny}, nx=nx,
                         feedthrough=feedthrough)
        self.nu = nu
        self.ny = ny
        self.nd = nd

    @abstractmethod
    def forward(self, u, x, d=None):
        """
            Forward pass of the model. This method should be implemented by subclasses.
            Args:
                u (Tensor): Input tensor.
                x (Tensor): State tensor.
                d (Tensor, optional): Disturbance tensor. Defaults to None.
    
            Returns:
                Tensor: Output tensor.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def __repr__(self):
        """
            String representation of the model.
        """
        if hasattr(self, 'nd'):
            return f"{self.__class__.__name__}(nu={self.nu}, ny={self.ny}, nx={self.nx}, nd={self.nd})"
        else:
            return f"{self.__class__.__name__}(nu={self.nu}, ny={self.ny}, nx={self.nx})"
    
    def __str__(self):
        """
            String representation of the model.
        """
        return self.__repr__()

    @abstractmethod
    def _frame(self) -> Tuple[Tensor, ...]:
        """
            This methods is the junsction from parameter space to weights space
            for non-parameterized models it is the identity operator.
        """
        pass 

    @abstractmethod
    def _right_inverse(self, *args, **kwargs):
        """
            From given weights that belongs to the manifold we initialize the parameter space
        """
        pass

    @abstractmethod
    def init_weights_(self, *args, **kwargs):
        """
            This method enables to initialize the weights it is both a wrapper for irght_inverse and
            the not parameterized weights of the module
        """
        pass

    @abstractmethod
    def clone(self):
        """
            Clone the model, it has to be implemented to be compliant with simulator classes.
        """
        raise NotImplementedError("State-space models must implement a clone method")

    def dynamics(self, inputs: Dict[str, Tensor], x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Adapter exposing the model to the diagram engine through named ports.

        Maps the ``forward(u, x, d) -> (dx, y)`` API onto the
        :meth:`~ctrlnmod.blocks.base.Block.dynamics` contract.
        """
        u = inputs["u"]
        d = inputs.get("d", None)
        dx, y = self.forward(u, x, d)
        return dx, {"y": y}

    def eval_output(self, x: Tensor, d: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Output of a strictly-proper model from its state alone (``y = g(x)``).

        Used by the diagram engine to break feedback loops: a strictly-proper model
        (``feedthrough=False``) produces its output before its input is known. The
        input ``u`` is irrelevant to ``y`` here, so a zero placeholder is passed.
        """
        u0 = x.new_zeros((x.shape[0], self.nu))
        if self.nd is not None and d is None:
            d = x.new_zeros((x.shape[0], self.nd))
        _, y = self.forward(u0, x, d)
        return {"y": y}

    # get_config / from_config are inherited from Block (recursive over Block attrs).