"""Generic neural feedback controllers (single input port -> single output port).

Unlike the feedback-linearizing controllers, :class:`NeuralController` is a plain
learnable map ``u = net(e)`` from an error/measurement port to a control port. It is
the natural Q-filter for Internal Model Control and a convenient policy block for
generic feedback loops. Set ``lip`` to obtain a certified Lipschitz bound (the net
is then an :class:`~ctrlnmod.models.feedforward.LBDN`).
"""

from typing import List, Optional, Union

import torch
from torch import Tensor

from ctrlnmod.controllers.base import Controller
from ctrlnmod.models.feedforward import FFNN, LBDN
from ctrlnmod.utils import parse_act_f


class NeuralController(Controller):
    """A static neural controller mapping one input port to one output port.

    Args:
        n_in: Input (e.g. error) width.
        n_out: Control output width.
        hidden_layers: Hidden-layer widths of the network.
        act_f: Activation function name. Defaults to ``'tanh'``.
        in_port: Name of the (single) input port. Defaults to ``'e'``.
        out_port: Name of the (single) output port. Defaults to ``'u'``.
        lip: Optional Lipschitz upper bound; if set, the net is an ``LBDN``.
    """

    def __init__(self, n_in: int, n_out: int, hidden_layers: List[int] = (32, 32),
                 act_f: str = 'tanh', in_port: str = 'e', out_port: str = 'u',
                 lip: Optional[Union[float, Tensor]] = None):
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_layers = list(hidden_layers)
        self.act_f_str = act_f
        self.in_port = in_port
        self.out_port = out_port
        self.lip = lip
        super().__init__(in_ports={in_port: n_in}, out_ports={out_port: n_out},
                         nx=0, feedthrough=True)
        self.act_f = parse_act_f(act_f)
        if lip is None:
            self.net = FFNN(n_in=n_in, n_out=n_out, hidden_layers=self.hidden_layers, act_f=act_f)
        else:
            self.net = LBDN(n_in=n_in, n_out=n_out, hidden_layers=self.hidden_layers,
                            act_f=act_f, scale=lip)

    def forward(self, e: Tensor) -> Tensor:
        return self.net(e)

    def evaluate(self, inputs, batch_size: int = 1):
        return {self.out_port: self.net(inputs[self.in_port])}

    def init_weights_(self, init=torch.nn.init.kaiming_uniform_):
        if hasattr(self.net, "init_weights_"):
            self.net.init_weights_()

    def clone(self) -> "NeuralController":
        cloned = NeuralController(
            n_in=self.n_in, n_out=self.n_out, hidden_layers=self.hidden_layers,
            act_f=self.act_f_str, in_port=self.in_port, out_port=self.out_port,
            lip=self.lip,
        )
        cloned.net = self.net.clone()
        return cloned
