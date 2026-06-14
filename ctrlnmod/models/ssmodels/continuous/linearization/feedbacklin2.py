import torch
from torch import Tensor
from typing import Optional, Union, List, Tuple
from ctrlnmod.models.ssmodels.base import SSModel
from ctrlnmod.utils import parse_act_f

# The feedback-linearizing controllers now live in ``ctrlnmod.controllers`` as
# first-class ``Controller`` blocks. They are re-exported here for backwards
# compatibility with code/tests importing them from this module.
from ctrlnmod.controllers.feedback_lin import (
    ILOFController,
    BetaILOFController,
    ILSFController,
    BetaILSFController,
)

__all__ = [
    "FLNSSM",
    "ILOFController",
    "BetaILOFController",
    "ILSFController",
    "BetaILSFController",
]

"""
This module implements architectures for learning approximately linearizing
controllers for a system, given I-O pairs of data. The learnable controllers
(``ILOFController`` etc.) are combined with a linear reference model so that the
closed loop matches a prescribed linear behaviour.
"""


class FLNSSM(SSModel):
    """
    A Feedforward Linearizable Neural Network State-Space Model (FLNSSM) that learns a linearizing controller
    for a given system. It can be an Output Feedback or State Feedback controller, depending on the inputs provided.

    Structurally this is the feedback interconnection of a learnable
    :class:`~ctrlnmod.controllers.base.Controller` with a linear reference model:
    the controller maps ``(u, y_or_x, d)`` to a virtual input ``v`` that drives the
    linear model, whose output is the model output. It can equivalently be built
    with :class:`~ctrlnmod.diagram.interconnections.FeedbackInterconnection`; this
    class keeps a direct, allocation-light ``forward`` for the common case.

    Args:
        nu (int): Number of inputs.
        ny (int): Number of outputs.
        nx (int): Number of states.
        linear_model (SSModel): Linear model used as the reference model for the closed-loop system.
        controller_type (str, optional): Type of controller to learn ('output_feedback' or 'state_feedback'). Defaults to 'output_feedback'.
        nd (Optional[int]): Number of disturbances/exogenous signals. Defaults to None.
        hidden_layers (List[int]): List of integers specifying the number of neurons in each hidden layer.
        act_f (str, optional): Activation function to use in the hidden layers. Defaults to 'relu'.
        lip (Optional[Union[float, Tensor]], optional): Lipschitz constant upper bound for the MLPs. Defaults to None.
    """
    def __init__(self, nu: int, ny: int, nx: int, linear_model: SSModel,
                 controller_type: str = 'output_feedback',
                 nd: Optional[int] = None,
                 hidden_layers: List[int] = [64, 64], act_f: str = 'relu',
                 lip: Optional[Union[float, Tensor]] = None):
        super(FLNSSM, self).__init__(nu=nu, ny=ny, nx=nx, nd=nd)

        self.hidden_layers = hidden_layers
        self.act_f = parse_act_f(act_f)
        self.act_f_str = act_f

        self.linear_model = linear_model
        self.nd = nd
        self.controller_type = controller_type

        self._frame_cache.register_child(self.linear_model._frame_cache)

        assert self.linear_model.nd == self.nd, \
            f"Linear model's nd ({self.linear_model.nd}) must match the model's nd ({self.nd})"
        assert self.linear_model.nu == self.nu, \
            f"Linear model's nu ({self.linear_model.nu}) must match the model's nu ({self.nu})"
        assert self.linear_model.ny == self.ny, \
            f"Linear model's ny ({self.linear_model.ny}) must match the model's ny ({self.ny})"

        if controller_type == 'output_feedback':
            self.controller = ILOFController(nu=nu, ny=ny, hidden_layers=hidden_layers, act_f=act_f, nd=nd, lip=lip)
        elif controller_type == 'state_feedback':
            self.controller = ILSFController(nu=nu, nx=nx, hidden_layers=hidden_layers, act_f=act_f, nd=nd, lip=lip)
        elif controller_type == 'beta_output_feedback':
            self.controller = BetaILOFController(nu=nu, ny=ny, hidden_layers=hidden_layers, act_f=act_f, nd=nd, lip=lip)
        elif controller_type == 'beta_state_feedback':
            self.controller = BetaILSFController(nu=nu, nx=nx, hidden_layers=hidden_layers, act_f=act_f, nd=nd, lip=lip)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}. Supported types are 'output_feedback', 'state_feedback', 'beta_output_feedback', and 'beta_state_feedback'.")

    def forward(self, u, x, d=None):
        """
        Forward pass of the FLNSSM model.

        Args:
            u (Tensor): Input tensor of shape (batch_size, nu).
            x (Tensor): State tensor of shape (batch_size, nx).
            d (Optional[Tensor]): Disturbance tensor of shape (batch_size, nd). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, ny), which is the output of the linear model.
        """
        if 'output' in self.controller_type:  # Output feedback
            # Pre computing the current y
            if 'Exo' in self.linear_model.__class__.__name__:
                _, _, C, _ = self._frame()
            else:
                _, _, C = self._frame()
            y = x @ C.T
            v = self.controller(u, y, d)
        else:  # State feedback
            v = self.controller(u, x, d)
        dx, y = self.linear_model(v, x, d)
        return dx, y

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  nu={self.nu}, ny={self.ny}, nx={self.nx}, nd={self.nd},\n"
            f"  controller={repr(self.controller)},\n"
            f"  linear_model={repr(self.linear_model), '    '}\n"
            f")"
        )

    def __str__(self):
        """String representation of the FLNSSM model."""
        return "FLNSSM Model:\n" + self.__repr__()

    def _frame(self) -> Tuple[Tensor, ...]:
        """Returns the weights from parameterized modules."""
        return self.linear_model._frame()

    def _right_inverse(self):
        pass  # Every initialisation is done through init_weights_ of the linear model and the controller

    def init_weights_(self, *args, **kwargs):
        """Initialize the weights of the linear model and the controller."""
        torch.nn.init.zeros_(self.controller.alpha.layers.output.weight)

    def clone(self) -> 'FLNSSM':
        """Clone the FLNSSM model."""
        cloned_model = FLNSSM(nu=self.nu, ny=self.ny, nx=self.nx, linear_model=self.linear_model.clone(),
                              controller_type=self.controller_type,
                              nd=self.nd, hidden_layers=self.hidden_layers, act_f=self.act_f_str)

        cloned_model.linear_model = self.linear_model.clone()
        cloned_model.controller = self.controller.clone()
        return cloned_model
