"""Base class for controllers in the block-diagram layer.

A :class:`Controller` is a feedback :class:`~ctrlnmod.blocks.base.Block`. Most
controllers are *static* algebraic feedback maps (``nx == 0``) producing a control
signal from a reference, a measurement (output or state) and optional disturbances,
but the abstraction also accommodates dynamic controllers (``nx > 0``) such as
observers or integral actions.

Promoting controllers to a first-class block (rather than a bare ``nn.Module``)
gives them named ports, a uniform ``clone`` / config lifecycle and lets them be
wired into any :class:`~ctrlnmod.diagram.diagram.Diagram` (IMC, feedback, MPC ...).
"""

from abc import abstractmethod
from typing import Dict

from ctrlnmod.blocks.base import Block


class Controller(Block):
    """Abstract base class for feedback controllers.

    Subclasses declare their input ports (e.g. ``u``/reference, ``y`` or ``x``,
    optional ``d``) and a control output port (``v``), and implement
    :meth:`evaluate` (static controllers) or :meth:`dynamics` (dynamic controllers).
    """

    def __init__(self, in_ports: Dict[str, int], out_ports: Dict[str, int],
                 nx: int = 0, feedthrough: bool = True) -> None:
        super().__init__(in_ports=in_ports, out_ports=out_ports, nx=nx,
                         feedthrough=feedthrough)

    @abstractmethod
    def clone(self) -> "Controller":
        raise NotImplementedError("Controllers must implement a clone method")
