import torch
import torch.nn as nn
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Union, Dict, Any, List, Tuple
from ctrlnmod.layers import BetaLayer
from ctrlnmod.utils import FrameCacheManager
from ctrlnmod.models.ssmodels.base import SSModel
import ctrlnmod.models.ssmodels.continuous.linear as lin
from ctrlnmod.models.feedforward import FFNN, LBDN
from ctrlnmod.utils import FrameCacheManager, parse_act_f
from abc import ABC, abstractmethod
""" 

This module implements different architectures for learning approximately linearizing controllers for a system, given I-O pairs of data.

"""


"""
    First we define the inverse of linearizing controllers, which will be included into the network's architecture
    in conjunction with a linear part which will be the target reference model for the closed-loop system.
    The linearizing controllers are defined as follows:
    - ILOFController: Inverse of a Linearizing Output Feedback controller :
        .. math::
            v = u + \alpha(y)
        or
        .. math::
            v = u + \alpha(y, d)
        if disturbances/exogenous signals are present.
    - ILSFController: Inverse of a Linearizing State Feedback controller :
        .. math::
            v = u + \alpha(x)
        or
        .. math::
            v = u + \alpha(x, d)
        if disturbances/exogenous signals are present.

    - BetaILOFController: Inverse of a Linearizing Output Feedback controller with a Beta decoupling matrix function :
        .. math::
            v = \beta(y)(u + \alpha(y))
        with :math:`\beta` is invertible. 
    - BetaILSFController: Inverse of a Linearizing State Feedback controller with a Beta decoupling matrix function :
        .. math::
            v = \beta(x)(u + \alpha(x))
        with :math:`\beta` is invertible.
"""
class ILOFController(nn.Module):
    """
    Inverse of a Linearizing Output Feedback (ILOF) controller : 
    .. math::
        v = u + \alpha(y)
    or
    .. math::
        v = u + \alpha(y, d)
    if disturbances are present.

    :math:`\alpha` is a function approximating the linearizing controller.
    and v is the new virtual input to the closed-loop system.

    Args:
        nu (int): Number of inputs.
        ny (int): Number of outputs.
        hidden_layers (List[int]): List of integers specifying the number of neurons in each hidden layer
        act_f (str, optional): Activation function to use in the hidden layers. Defaults to 'relu'.
        nd (Optional[int], optional): Number of disturbances/exogenous signals. Defaults to
        lip (optional): Lipschitz constant upper bound for the controller. Defaults to None.
    """
    def __init__(self, nu: int, ny: int, hidden_layers: List[int], 
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]]=None):
        super(ILOFController, self).__init__()
        if not isinstance(nu, int) or nu < 0:
            raise ValueError("nu must be a non-negative integer")
        if not isinstance(ny, int) or ny < 0:
            raise ValueError("ny must be a non-negative integer")
        if nd is not None and (not isinstance(nd, int) or nd < 0):
            raise ValueError("nd must be a non-negative integer or None")
        
        self.nu = nu
        self.ny = ny
        self.hidden_layers = hidden_layers

        self.inversed = True  # A flag to specify the forward pass is the inverse of the actual controller
        if nd is not None:
            self.nd = nd
            n_inputs = ny + nd
        else:
            n_inputs =  ny
        self.act_f = parse_act_f(act_f)
        self.act_f_str = act_f

        if lip is None:
            self.alpha = FFNN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f)
        else:
            self.alpha = LBDN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f, scale=lip)

    def forward(self, u: Tensor, y: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the ILOF controller.

        Args:
            u (Tensor): Input tensor of shape (batch_size, nu).
            y (Tensor): Output tensor of shape (batch_size, ny).
            d (Optional[Tensor]): Disturbance tensor of shape (batch_size, nd). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, nu), which is the new virtual input.
        """
        if d is not None:
            inputs = torch.cat((y, d), dim=-1)
        else:
            inputs = y
        
        alpha_y = self.alpha(inputs)
        return u + alpha_y

    def clone(self) -> 'ILOFController':
        """
        Clone the ILOF controller.

        Returns:
            ILOFController: A new instance of the ILOF controller with the same parameters.
        """
        cloned_controller = ILOFController(nu=self.nu, ny=self.ny, hidden_layers=self.hidden_layers, 
                                           act_f=self.act_f_str, nd=self.nd)
        cloned_controller.alpha = self.alpha.clone()
        return cloned_controller
    
    def inverse(self, v, y_hat, d_hat):
        """
            Inverse of the forward pass : it gives the linearizing control law with a new entry v.
        """

        alpha = self.alpha(torch.cat((y_hat, d_hat), dim=1))
        # print(f"Alpha value: {alpha} ")
        return v - alpha
    
class BetaILOFController(ILOFController):
    r"""
    Inverse of a Linearizing Output Feedback (ILOF) controller with a Beta decoupling matrix function.

    .. math::
        v = \beta(y)(u + alpha(y))
    
    with :math:`\beta` is invertible.

    """

    def __init__(self, nu: int, ny: int, hidden_layers: List[int], 
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]]=None):
        """
        Args:
            nu (int): Number of inputs.
            ny (int): Number of outputs.
            hidden_layers (List[int]): List of integers specifying the number of neurons in each hidden layer.
            act_f (str, optional): Activation function to use in the hidden layers. Defaults to 'relu'.
            nd (Optional[int], optional): Number of disturbances/exogenous signals. Defaults to None.
            lip (optional): Lipschitz constant upper bound for the MLP. Defaults to None.
        """
        super(BetaILOFController, self).__init__(nu, ny, hidden_layers, act_f, nd, lip)
        if nd is not None:
            n_inputs = ny + nd
        else:
            n_inputs = ny
        self.act_f = parse_act_f(act_f)
        self.beta = BetaLayer(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=self.act_f)

        self.inversed = True  # A flag to specify the forward pass is the inverse of the actual controller
    
    def forward(self, u: Tensor, y: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Beta ILOF controller.

        Args:
            u (Tensor): Input tensor of shape (batch_size, nu).
            y (Tensor): Output tensor of shape (batch_size, ny).
            d (Optional[Tensor]): Disturbance tensor of shape (batch_size, nd). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, nu), which is the new virtual input.
        """
        if d is not None:
            inputs = torch.cat((y, d), dim=-1)
        else:
            inputs = y
        
        beta_y = self.beta(inputs)
        alpha_y = self.alpha(inputs)
        return torch.matmul(beta_y, (u + alpha_y).unsqueeze(-1)).squeeze(-1)
    
    def clone(self) -> 'BetaILOFController':
        """
        Clone the Beta ILOF controller.

        Returns:
            BetaILOFController: A new instance of the Beta ILOF controller with the same parameters.
        """
        cloned_controller = BetaILOFController(nu=self.nu, ny=self.ny, hidden_layers=self.hidden_layers, 
                                               act_f=self.act_f_str, nd=self.nd)
        cloned_controller.alpha = self.alpha.clone()
        cloned_controller.beta = self.beta.clone()
        return cloned_controller
    
    def inverse(self, v, y_hat, d_hat):
        raise NotImplementedError("TODO")
    
class ILSFController(nn.Module):
    """
    Inverse of a Linearizing State Feedback (ILSF) controller : 
    .. math::
        v = u + \alpha(x)
    or
    .. math::
        v = u + \alpha(x, d)
    if disturbances are present.

    :math:`\alpha` is a function approximating the linearizing controller.
    and v is the new virtual input to the closed-loop system.

    Args:
        nu (int): Number of inputs.
        nx (int): Number of states.
        hidden_layers (List[int]): List of integers specifying the number of neurons in each hidden layer
        act_f (str, optional): Activation function to use in the hidden layers. Defaults to 'relu'.
        nd (Optional[int], optional): Number of disturbances/exogenous signals. Defaults to None.
        lip (optional): Lipschitz constant upper bound for the controller. Defaults to None.
    """
    def __init__(self, nu: int, nx: int, hidden_layers: List[int], 
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]]=None):
        super(ILSFController, self).__init__()
        if not isinstance(nu, int) or nu < 0:
            raise ValueError("nu must be a non-negative integer")
        if not isinstance(nx, int) or nx < 0:
            raise ValueError("nx must be a non-negative integer")
        if nd is not None and (not isinstance(nd, int) or nd < 0):
            raise ValueError("nd must be a non-negative integer or None")
        
        self.nu = nu
        self.nx = nx
        self.hidden_layers = hidden_layers
        self.inversed = True  # A flag to specify the forward pass is the inverse of the actual controller


        if nd is not None:
            self.nd = nd
            n_inputs = nx + nd
        else:
            n_inputs =  nx
        self.act_f, self.act_f_str = parse_act_f(act_f), act_f

        if lip is None:
            self.alpha = FFNN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f)
        else:
            self.alpha = LBDN(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=act_f, scale=lip)

    def forward(self, u: Tensor, x: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the ILSF controller.

        Args:
            u (Tensor): Input tensor of shape (batch_size, nu).
            x (Tensor): State tensor of shape (batch_size, nx).
            d (Optional[Tensor]): Disturbance tensor of shape (batch_size, nd). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, nu), which is the new virtual input.
        """
        if d is not None:
            inputs = torch.cat((x, d), dim=-1)
        else:
            inputs = x
        
        alpha_x = self.alpha(inputs)
        return u + alpha_x
    

    def clone(self) -> 'ILSFController':
        """
        Clone the ILSF controller.

        Returns:
            ILSFController: A new instance of the ILSF controller with the same parameters.
        """
        cloned_controller = ILSFController(nu=self.nu, nx=self.nx, hidden_layers=self.hidden_layers, 
                                           act_f=self.act_f_str, nd=self.nd)
        cloned_controller.alpha = self.alpha.clone()
        return cloned_controller
    
    def inverse(self, v, x, d=None):
        if d is not None:
            inputs = torch.cat((x, d), dim=-1)
        else:
            inputs = x
        
        alpha_x = self.alpha(inputs)
        return v - alpha_x
    
class BetaILSFController(ILSFController):
    r"""
    Inverse of a Linearizing State Feedback (ILSF) controller with a Beta decoupling matrix function.

    .. math::
        v = \beta(x)(u + alpha(x))
    
    with :math:`\beta` is invertible.

    """

    def __init__(self, nu: int, nx: int, hidden_layers: List[int], 
                 act_f: str = 'relu', nd: Optional[int] = None,
                 lip: Optional[Union[float, Tensor]]=None):
        """
        Args:
            nu (int): Number of inputs.
            nx (int): Number of states.
            hidden_layers (List[int]): List of integers specifying the number of neurons in each hidden layer.
            act_f (str, optional): Activation function to use in the hidden layers. Defaults to 'relu'.
            nd (Optional[int], optional): Number of disturbances/exogenous signals. Defaults to None.
            lip (optional): Lipschitz constant upper bound for the MLP. Defaults to None.
        """
        super(BetaILSFController, self).__init__(nu, nx, hidden_layers, act_f, nd, lip)
        if nd is not None:
            n_inputs = nx + nd
        else:
            n_inputs = nx
        self.act_f = parse_act_f(act_f)
        self.beta = BetaLayer(n_in=n_inputs, n_out=nu, hidden_layers=hidden_layers, act_f=self.act_f)

    def forward(self, u: Tensor, x: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Beta ILSF controller.

        Args:
            u (Tensor): Input tensor of shape (batch_size, nu).
            x (Tensor): State tensor of shape (batch_size, nx).
            d (Optional[Tensor]): Disturbance tensor of shape (batch_size, nd). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, nu), which is the new virtual input.
        """
        if d is not None:
            inputs = torch.cat((x, d), dim=-1)
        else:
            inputs = x
        
        beta_x = self.beta(inputs)
        alpha_x = self.alpha(inputs)
        return torch.matmul(beta_x , (u + alpha_x).unsqueeze(-1)).squeeze(-1)
    
    def clone(self) -> 'BetaILSFController':
        """
        Clone the Beta ILSF controller.

        Returns:
            BetaILSFController: A new instance of the Beta ILSF controller with the same parameters.
        """
        cloned_controller = BetaILSFController(nu=self.nu, nx=self.nx, hidden_layers=self.hidden_layers, 
                                               act_f=self.act_f_str, nd=self.nd)
        cloned_controller.alpha = self.alpha.clone()
        cloned_controller.beta = self.beta.clone()
        return cloned_controller

"""
    In this section, we now build the linearizable neural network architectures
    which will be used to learn the linearizing controllers.
"""


class FLNSSM(SSModel):
    """
    A Feedforward Linearizable Neural Network State-Space Model (FLNSSM) that learns a linearizing controller
    for a given system. It can be an Output Feedback or State Feedback controller, depending on the inputs provided.

    Args:
        nu (int): Number of inputs.
        ny (int): Number of outputs.
        nx (int): Number of states.
        linear_model (Optional[SSModel], optional): Linear model type to use as a reference model for the closed-loop system. Defaults to None.
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
        else:  # State feedback
            v = self.controller(u, x, d)
        dx, y  = self.linear_model(v, x, d)
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
        """
        String representation of the FLNSSM model.
        """
        return "FLNSSM Model:\n" + self.__repr__()
    
    def _frame(self) -> Tuple[Tensor, ...]:
        """
        Returns the weights from parameterized modules.

        Returns:
            Tuple[Tensor, ...]: A tuple containing the frame tensors.
        """
        return self.linear_model._frame()
    
    def _right_inverse(self):
        pass  # Every initialisation must be done through the init_weights_ method of the linear model and the controller

    def init_weights_(self, *args, **kwargs):
        """
        Initialize the weights of the linear model and the controller.

        Args:
            *args: Positional arguments for the linear model and controller.
            **kwargs: Keyword arguments for the linear model and controller.
        """
        torch.nn.init.zeros_(self.controller.alpha.layers.output.weight)

    
    def clone(self) -> 'FLNSSM':
        """
        Clone the FLNSSM model.

        Returns:
            FLNSSM: A new instance of the FLNSSM model with the same parameters.
        """
        cloned_model = FLNSSM(nu=self.nu, ny=self.ny, nx=self.nx, linear_model=self.linear_model.clone(),
                             controller_type=self.controller_type,
                             nd=self.nd, hidden_layers=self.hidden_layers, act_f=self.act_f_str)

        cloned_model.linear_model = self.linear_model.clone()
        cloned_model.controller = self.controller.clone()
        return cloned_model