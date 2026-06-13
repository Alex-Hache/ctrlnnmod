r"""Approximate feedback-linearizing controllers.

These controllers learn an (inverse) linearizing control law from input/output (or
input/state) data, to be combined with a linear reference model in
:class:`~ctrlnmod.models.ssmodels.continuous.linearization.feedbacklin2.FLNSSM`.

They were historically plain ``nn.Module``s defined alongside ``FLNSSM``; they are
now first-class :class:`~ctrlnmod.controllers.base.Controller` blocks with named
ports, so they can also be dropped into a generic
:class:`~ctrlnmod.diagram.diagram.Diagram`. The control laws are unchanged:

* ``ILOFController``   : :math:`v = u + \alpha(y)`            (output feedback)
* ``ILSFController``   : :math:`v = u + \alpha(x)`            (state feedback)
* ``BetaILOFController``: :math:`v = \beta(y)\,(u + \alpha(y))`
* ``BetaILSFController``: :math:`v = \beta(x)\,(u + \alpha(x))`
"""

import torch
from torch import Tensor
from typing import List, Optional, Union

from ctrlnmod.controllers.base import Controller
from ctrlnmod.layers import BetaLayer
from ctrlnmod.models.feedforward import FFNN, LBDN
from ctrlnmod.utils import parse_act_f


class ILOFController(Controller):
    r"""Inverse linearizing **output** feedback controller :math:`v = u + \alpha(y)`.

    Args:
        nu: Number of (virtual) inputs.
        ny: Number of outputs.
        hidden_layers: Hidden layer widths of the :math:`\alpha` network.
        act_f: Activation function name. Defaults to ``'relu'``.
        nd: Number of disturbances/exogenous signals (or ``None``).
        lip: Optional Lipschitz upper bound; if set, :math:`\alpha` is an ``LBDN``.
    """

    def __init__(self, nu: int, ny: int, hidden_layers: List[int],
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]] = None):
        if not isinstance(nu, int) or nu < 0:
            raise ValueError("nu must be a non-negative integer")
        if not isinstance(ny, int) or ny < 0:
            raise ValueError("ny must be a non-negative integer")
        if nd is not None and (not isinstance(nd, int) or nd < 0):
            raise ValueError("nd must be a non-negative integer or None")

        in_ports = {"u": nu, "y": ny}
        if nd is not None:
            in_ports["d"] = nd
        super().__init__(in_ports=in_ports, out_ports={"v": nu},
                         nx=0, feedthrough=True)

        self.nu = nu
        self.ny = ny
        self.hidden_layers = hidden_layers
        self.nd = nd
        self.lip = lip
        self.inversed = True  # forward pass realises the inverse controller
        n_inputs = ny + (nd if nd is not None else 0)
        self.act_f = parse_act_f(act_f)
        self.act_f_str = act_f

        if lip is None:
            self.alpha = FFNN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f)
        else:
            self.alpha = LBDN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f, scale=lip)

    def forward(self, u: Tensor, y: Tensor, d: Optional[Tensor] = None) -> Tensor:
        inputs = torch.cat((y, d), dim=-1) if d is not None else y
        return u + self.alpha(inputs)

    def evaluate(self, inputs, batch_size: int = 1):
        return {"v": self.forward(inputs["u"], inputs["y"], inputs.get("d", None))}

    def inverse(self, v, y_hat, d_hat):
        """Linearizing control law for a new virtual input ``v``."""
        alpha = self.alpha(torch.cat((y_hat, d_hat), dim=1))
        return v - alpha

    def clone(self) -> "ILOFController":
        cloned = ILOFController(nu=self.nu, ny=self.ny, hidden_layers=self.hidden_layers,
                                act_f=self.act_f_str, nd=self.nd, lip=self.lip)
        cloned.alpha = self.alpha.clone()
        return cloned


class BetaILOFController(ILOFController):
    r"""Output feedback with a decoupling matrix: :math:`v = \beta(y)(u + \alpha(y))`."""

    def __init__(self, nu: int, ny: int, hidden_layers: List[int],
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]] = None):
        super().__init__(nu, ny, hidden_layers, act_f, nd, lip)
        n_inputs = ny + (nd if nd is not None else 0)
        self.beta = BetaLayer(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=self.act_f)

    def forward(self, u: Tensor, y: Tensor, d: Optional[Tensor] = None) -> Tensor:
        inputs = torch.cat((y, d), dim=-1) if d is not None else y
        beta_y = self.beta(inputs)
        alpha_y = self.alpha(inputs)
        return torch.matmul(beta_y, (u + alpha_y).unsqueeze(-1)).squeeze(-1)

    def inverse(self, v, y_hat, d_hat):
        raise NotImplementedError("TODO")

    def clone(self) -> "BetaILOFController":
        cloned = BetaILOFController(nu=self.nu, ny=self.ny, hidden_layers=self.hidden_layers,
                                    act_f=self.act_f_str, nd=self.nd, lip=self.lip)
        cloned.alpha = self.alpha.clone()
        cloned.beta = self.beta.clone()
        return cloned


class ILSFController(Controller):
    r"""Inverse linearizing **state** feedback controller :math:`v = u + \alpha(x)`.

    Args:
        nu: Number of (virtual) inputs.
        nx: Number of states.
        hidden_layers: Hidden layer widths of the :math:`\alpha` network.
        act_f: Activation function name. Defaults to ``'relu'``.
        nd: Number of disturbances/exogenous signals (or ``None``).
        lip: Optional Lipschitz upper bound; if set, :math:`\alpha` is an ``LBDN``.
    """

    def __init__(self, nu: int, nx: int, hidden_layers: List[int],
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]] = None):
        if not isinstance(nu, int) or nu < 0:
            raise ValueError("nu must be a non-negative integer")
        if not isinstance(nx, int) or nx < 0:
            raise ValueError("nx must be a non-negative integer")
        if nd is not None and (not isinstance(nd, int) or nd < 0):
            raise ValueError("nd must be a non-negative integer or None")

        in_ports = {"u": nu, "x": nx}
        if nd is not None:
            in_ports["d"] = nd
        super().__init__(in_ports=in_ports, out_ports={"v": nu},
                         nx=0, feedthrough=True)

        self.nu = nu
        self.nx = nx
        self.hidden_layers = hidden_layers
        self.nd = nd
        self.lip = lip
        self.inversed = True
        n_inputs = nx + (nd if nd is not None else 0)
        self.act_f, self.act_f_str = parse_act_f(act_f), act_f

        if lip is None:
            self.alpha = FFNN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f)
        else:
            self.alpha = LBDN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f, scale=lip)

    def forward(self, u: Tensor, x: Tensor, d: Optional[Tensor] = None) -> Tensor:
        inputs = torch.cat((x, d), dim=-1) if d is not None else x
        return u + self.alpha(inputs)

    def evaluate(self, inputs, batch_size: int = 1):
        return {"v": self.forward(inputs["u"], inputs["x"], inputs.get("d", None))}

    def inverse(self, v, x, d=None):
        inputs = torch.cat((x, d), dim=-1) if d is not None else x
        return v - self.alpha(inputs)

    def clone(self) -> "ILSFController":
        cloned = ILSFController(nu=self.nu, nx=self.nx, hidden_layers=self.hidden_layers,
                                act_f=self.act_f_str, nd=self.nd, lip=self.lip)
        cloned.alpha = self.alpha.clone()
        return cloned


class BetaILSFController(ILSFController):
    r"""State feedback with a decoupling matrix: :math:`v = \beta(x)(u + \alpha(x))`."""

    def __init__(self, nu: int, nx: int, hidden_layers: List[int],
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]] = None):
        super().__init__(nu, nx, hidden_layers, act_f, nd, lip)
        n_inputs = nx + (nd if nd is not None else 0)
        self.beta = BetaLayer(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=self.act_f)

    def forward(self, u: Tensor, x: Tensor, d: Optional[Tensor] = None) -> Tensor:
        inputs = torch.cat((x, d), dim=-1) if d is not None else x
        beta_x = self.beta(inputs)
        alpha_x = self.alpha(inputs)
        return torch.matmul(beta_x, (u + alpha_x).unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "BetaILSFController":
        cloned = BetaILSFController(nu=self.nu, nx=self.nx, hidden_layers=self.hidden_layers,
                                    act_f=self.act_f_str, nd=self.nd, lip=self.lip)
        cloned.alpha = self.alpha.clone()
        cloned.beta = self.beta.clone()
        return cloned
