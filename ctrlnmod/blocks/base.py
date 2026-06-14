"""Base abstractions for the block-diagram ("Simulink") layer of ctrlnmod.

A :class:`Block` is the common root for every component that can be wired into a
:class:`~ctrlnmod.diagram.diagram.Diagram`. It generalises the existing
:class:`~ctrlnmod.models.ssmodels.base.SSModel` so that both dynamical systems
(plants, internal models) and algebraic maps (controllers, references, gains)
share a single interface with **named input/output ports**.

A block is either

* **static** (``nx == 0``): a pure algebraic map ``outputs = evaluate(inputs)``;
* **dynamic** (``nx > 0``): a state-space block ``(dx, outputs) = dynamics(inputs, x)``.

The ``feedthrough`` flag tells the diagram engine whether any output depends
*directly* on the inputs. It is used to order block evaluation and to detect
algebraic loops. Strictly-proper state-space models (``y = g(x)``) have
``feedthrough=False`` and therefore break algebraic loops.

The lifecycle methods (``clone``, ``get_config``/``from_config``, ``_frame``,
``_right_inverse``, ``init_weights_``) and the hierarchical
:class:`~ctrlnmod.utils.FrameCacheManager` mirror those already used throughout
the library, so existing parametrization/caching machinery keeps working.
"""

from abc import ABC, abstractmethod
from importlib import import_module
import inspect
from typing import Dict, Tuple

from torch import Tensor
from torch.nn import Module

from ctrlnmod.utils import FrameCacheManager


def get_fqn(obj) -> str:
    """Return the fully-qualified name ``module.ClassName`` of ``obj``."""
    return obj.__module__ + "." + obj.__class__.__name__


def import_class(fqn: str):
    """Import a class from its fully-qualified name ``module.ClassName``."""
    mod, cls = fqn.rsplit(".", 1)
    return getattr(import_module(mod), cls)


class Block(Module, ABC):
    """Common base class for all diagram blocks.

    Args:
        in_ports: Mapping ``port_name -> width`` for the inputs consumed by the block.
        out_ports: Mapping ``port_name -> width`` for the outputs produced by the block.
        nx: State dimension. ``0`` denotes a static (algebraic) block.
        feedthrough: Whether any output depends directly on the inputs. Defaults to
            ``True`` (the safe choice for static blocks). Strictly-proper dynamic
            blocks should set this to ``False`` so the engine can break algebraic loops.
    """

    def __init__(
        self,
        in_ports: Dict[str, int],
        out_ports: Dict[str, int],
        nx: int = 0,
        feedthrough: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(nx, int) or nx < 0:
            raise ValueError("nx must be a non-negative integer")
        self.in_ports: Dict[str, int] = dict(in_ports)
        self.out_ports: Dict[str, int] = dict(out_ports)
        # ``state_dim`` is the block's OWN state dimension and is what the diagram
        # engine relies on. ``self.nx`` is also set for convenience/back-compat, but
        # some subclasses (e.g. state-feedback controllers) repurpose ``self.nx`` to
        # denote a *plant* state size they read - hence the dedicated attribute.
        self._state_dim = nx
        self.nx = nx
        self.feedthrough = bool(feedthrough)
        self._frame_cache = FrameCacheManager()

    # ------------------------------------------------------------------ #
    # Port helpers
    # ------------------------------------------------------------------ #
    @property
    def state_dim(self) -> int:
        """The block's own state dimension (immune to ``self.nx`` reuse)."""
        return self._state_dim

    @property
    def is_dynamic(self) -> bool:
        """Whether the block carries state (``state_dim > 0``)."""
        return self._state_dim > 0

    def input_names(self):
        return list(self.in_ports.keys())

    def output_names(self):
        return list(self.out_ports.keys())

    # ------------------------------------------------------------------ #
    # Engine interface
    # ------------------------------------------------------------------ #
    def evaluate(self, inputs: Dict[str, Tensor], batch_size: int = 1) -> Dict[str, Tensor]:
        """Algebraic evaluation of a *static* block.

        Args:
            inputs: Mapping ``port_name -> tensor`` for every declared input port.
            batch_size: Running batch size, used by source blocks (no inputs) to
                broadcast their output.

        Returns:
            Mapping ``port_name -> tensor`` for every declared output port.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is dynamic; the diagram engine must call "
            "dynamics() instead of evaluate()."
            if self.is_dynamic
            else f"{type(self).__name__} must implement evaluate()."
        )

    def dynamics(self, inputs: Dict[str, Tensor], x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """State-space evaluation of a *dynamic* block.

        Args:
            inputs: Mapping ``port_name -> tensor`` for every declared input port.
            x: Current state, shape ``(batch, nx)``.

        Returns:
            A tuple ``(dx, outputs)`` where ``dx`` is the state derivative
            (continuous) or next state (discrete) and ``outputs`` maps output port
            names to tensors.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is static; the diagram engine must call "
            "evaluate() instead of dynamics()."
        )

    # ------------------------------------------------------------------ #
    # Lifecycle (parametrization friendly, mirrors SSModel)
    # ------------------------------------------------------------------ #
    def _frame(self) -> Tuple[Tensor, ...]:
        """Map parameter space to weight space. Identity (empty) by default."""
        return tuple()

    def _right_inverse(self, *args, **kwargs):
        """Initialise the parameter space from weights living on the manifold."""
        return None

    def init_weights_(self, *args, **kwargs):
        """Public weight-initialisation hook. No-op by default."""
        return None

    @abstractmethod
    def clone(self) -> "Block":
        """Return a deep copy of the block (required by simulators/diagrams)."""
        raise NotImplementedError("Blocks must implement a clone method")

    # ------------------------------------------------------------------ #
    # Trainability helpers
    # ------------------------------------------------------------------ #
    def freeze(self) -> "Block":
        """Freeze every parameter of the block (``requires_grad = False``)."""
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self) -> "Block":
        """Unfreeze every parameter of the block (``requires_grad = True``)."""
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    # ------------------------------------------------------------------ #
    # Serialization (recursive, same convention as the rest of the library)
    # ------------------------------------------------------------------ #
    def get_config(self) -> dict:
        """Return a serialisable config built from the ``__init__`` signature.

        Every ``__init__`` parameter must have a matching attribute on the
        instance. Nested :class:`Block` attributes are serialised recursively.
        """
        cls = self.__class__
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for name, _ in sig.parameters.items():
            if name == "self":
                continue
            if not hasattr(self, name):
                raise ValueError(f"Attribute '{name}' not found in {cls.__name__}")
            value = getattr(self, name)
            if isinstance(value, Block):
                value = value.get_config()
            kwargs[name] = value
        return {"class": get_fqn(self), "kwargs": kwargs}

    @classmethod
    def from_config(cls, config: dict) -> "Block":
        """Rebuild a block (and any nested blocks) from a :meth:`get_config` dict."""
        kwargs = config["kwargs"]
        for k, v in kwargs.items():
            if isinstance(v, dict) and "class" in v and "kwargs" in v:
                sub_cls = import_class(v["class"])
                kwargs[k] = sub_cls.from_config(v)
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_ports={self.in_ports}, "
            f"out_ports={self.out_ports}, nx={self.nx})"
        )
