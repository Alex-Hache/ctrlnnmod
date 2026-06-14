"""Primitive static blocks - the elementary building blocks of a diagram.

These mirror the most common Simulink primitives (Sum, Gain, Constant, Source,
Saturation). They are all *static* (``nx == 0``) and fully differentiable, so they
can be freely wired into a :class:`~ctrlnmod.diagram.diagram.Diagram` and trained.
"""

from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from ctrlnmod.blocks.base import Block


class Sum(Block):
    r"""Signed sum of several inputs of identical width.

    Computes ``out = sum_i sign_i * in_i``. With ``signs=('+', '-')`` and inputs
    named ``in0, in1`` this realises ``out = in0 - in1`` (a comparator/error node).

    Args:
        width: Common width of every input/output port.
        signs: Sequence of ``'+'``/``'-'`` (or ``+1``/``-1``) giving the number of
            inputs and their signs. Defaults to two positive inputs.
    """

    def __init__(self, width: int, signs: List[Union[str, int]] = ("+", "+")) -> None:
        self.width = width
        self.signs = list(signs)
        self._coeffs = [1.0 if s in ("+", 1, +1) else -1.0 for s in signs]
        in_ports = {f"in{i}": width for i in range(len(self.signs))}
        super().__init__(in_ports=in_ports, out_ports={"out": width},
                         nx=0, feedthrough=True)

    def evaluate(self, inputs: Dict[str, Tensor], batch_size: int = 1) -> Dict[str, Tensor]:
        out = None
        for i, coeff in enumerate(self._coeffs):
            term = coeff * inputs[f"in{i}"]
            out = term if out is None else out + term
        return {"out": out}

    def clone(self) -> "Sum":
        return Sum(width=self.width, signs=list(self.signs))


class Gain(Block):
    r"""Static (matrix or scalar) gain ``out = K @ in`` (or ``k * in``).

    Args:
        n_in: Input width.
        n_out: Output width. Defaults to ``n_in`` (scalar/square gain).
        gain: Initial gain. A scalar broadcasts; a matrix must be ``(n_out, n_in)``.
        trainable: Whether the gain is a learnable parameter.
    """

    def __init__(self, n_in: int, n_out: Optional[int] = None,
                 gain: Union[float, Tensor] = 1.0, trainable: bool = False) -> None:
        self.n_in = n_in
        self.n_out = n_out if n_out is not None else n_in
        self.gain = gain
        self.trainable = trainable
        super().__init__(in_ports={"in": n_in}, out_ports={"out": self.n_out},
                         nx=0, feedthrough=True)
        dtype = torch.get_default_dtype()
        if isinstance(gain, Tensor) and gain.ndim == 2:
            K = gain.detach().clone().to(dtype)
        else:
            K = float(gain) * torch.eye(self.n_out, self.n_in, dtype=dtype)
        if trainable:
            self.K = torch.nn.Parameter(K)
        else:
            self.register_buffer("K", K)

    def evaluate(self, inputs: Dict[str, Tensor], batch_size: int = 1) -> Dict[str, Tensor]:
        return {"out": inputs["in"] @ self.K.T}

    def clone(self) -> "Gain":
        copy = Gain(self.n_in, self.n_out, gain=self.gain, trainable=self.trainable)
        copy.load_state_dict(self.state_dict())
        return copy


class Constant(Block):
    """A constant source with no inputs, broadcast to the running batch size.

    Args:
        value: The constant tensor of shape ``(width,)`` (or a float for width 1).
    """

    def __init__(self, value: Union[float, Tensor]) -> None:
        val = torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(-1)
        self.value = val
        super().__init__(in_ports={}, out_ports={"out": val.shape[0]},
                         nx=0, feedthrough=False)
        self.register_buffer("_value", val)

    def evaluate(self, inputs: Dict[str, Tensor], batch_size: int = 1) -> Dict[str, Tensor]:
        return {"out": self._value.unsqueeze(0).expand(batch_size, -1)}

    def clone(self) -> "Constant":
        return Constant(self.value.clone())


class Saturation(Block):
    """Element-wise saturation ``out = clamp(in, low, high)``.

    Args:
        width: Port width.
        low: Lower bound (scalar).
        high: Upper bound (scalar).
    """

    def __init__(self, width: int, low: float = -1.0, high: float = 1.0) -> None:
        self.width = width
        self.low = low
        self.high = high
        super().__init__(in_ports={"in": width}, out_ports={"out": width},
                         nx=0, feedthrough=True)

    def evaluate(self, inputs: Dict[str, Tensor], batch_size: int = 1) -> Dict[str, Tensor]:
        return {"out": torch.clamp(inputs["in"], self.low, self.high)}

    def clone(self) -> "Saturation":
        return Saturation(self.width, self.low, self.high)
