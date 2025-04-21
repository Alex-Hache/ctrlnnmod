from .lbdn import FFNN, Fxu, LBDN, LipHx, Hx, LipFxu
from ...layers.layers import BetaLayer
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
import torch



class LinearizingController(nn.Module, ABC):

    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int, n_layers: int, 
                 beta: nn.Module = None, act_f: str = 'tanh', bias: bool = True,
                 lipschitz_bound: float = None):
        r"""Base class for linearzing controllers.
                u_{lin} = a(.) + beta(.)
        Args:
            n_inputs (int): Number of inputs for the controller.
            n_outputs (int): Number of outputs for the controller.
            beta (nn.Module, optional): Module that computes the beta decoupling matrix for feedback linearization.
                Defaults to None.
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.beta = beta
        self.lip = lipschitz_bound
        # Activation function setup
        if act_f.lower() == 'tanh':
            self.act_f = nn.Tanh()
        elif act_f.lower() == 'relu':
            self.act_f = nn.ReLU()
        else:
            raise NotImplementedError(f"Function {act_f} not implemented. Choose 'tanh' or 'relu'.")
        
        self.bias = bias
        if lipschitz_bound is not None:
            self.alpha = LBDN(n_inputs, n_hidden, n_outputs, scale=lipschitz_bound, act_f=self.act_f, n_hidden=n_layers, bias=self.bias)
        else:
            self.alpha = FFNN(n_inputs, n_hidden, n_outputs, self.act_f, n_layers, self.bias)

        if beta is not None:
            if self.lip is None:  # If we enforce an upper bound on the lipschitz constant of terms we scale the singular values of beta accordingly
                scale = 1
            else:
                scale = self.lip
            self.beta = BetaLayer(nu=n_outputs, nx=n_inputs, nh=n_hidden, act_f=self.act_f, scale=scale)
        else:
            self.beta = None

    def forward(self, inputs: Tensor, u: Tensor):

        assert inputs.shape[1] == self.n_inputs, f"Number of controller's input mismatch expected {self.n_inputs} found {inputs.shape[1]}"


        alpha = self.alpha(inputs)

        if self.beta is not None:
            beta = self.beta(inputs)
        else:
            beta = torch.eye(self.n_inputs).unsqueeze(0).expand(inputs.shape[0], self.n_inputs, self.n_inputs)
        
        v = torch.bmm(beta, u + alpha)

        return v
    
class OutputFeedbackLinearizingcontroller(LinearizingController):
    r"""Initialize an output linearizing controller.
            u_{lin} = \alpha(y,d) + \beta(y,d)v or u_{lin} = \alpha(y,d) + vif beta is set to None
        Args:
            ny (int): Number of outputs of your system (# of inputs for your controller).
            nu (int): Number of inputs of your system (# of outputs for your controller).
            nh (int): Number of hidden neurons per layer
            n_hidden_layers (int): Number of hidden layers
            beta (nn.Module, optional): Module that computes the beta decoupling matrix for feedback linearization.
                Defaults to None.
            nd (int): Number of measured disturbances the linearizing controller will absorb
                Defaults to 0
            lipschitz_bound (float): Upper bound on the Lipschitz constant for the neural controller parts
                Defaults to None
        """
    def __init__(self, ny: int, nu: int, nh: int, n_hidden_layers: int, beta: nn.Module = None, nd: int = 0, lipschitz_bound: float = None):
        super().__init__(ny + nd, nu, nh, n_hidden_layers, beta=beta, lipschitz_bound=lipschitz_bound)
        self.ny = ny
        self.nu = nu
        self.nd = nd

    def forward(self, y: Tensor, u: Tensor, d:Tensor = None):
        """
            Args:
                y (Tensor): system's measured outputs 
                    Size : [batch_size, ny]
                u (Tensor):system's control input
                    Size : [batch_size, nu]
                d (Tensor) : measured disturbances 
                    Size : [batch_size, nd]
                    Defaults to None
        """
        if d is not None:
            inputs = torch.cat(y, d, dim=1)
        else:
            inputs = y
        
        return super().forward(inputs=inputs, u=u)


class StateFeedbackLinearizingcontroller(LinearizingController):
    r"""Initialize a state-feedback linearizing controller.
            u_{lin} = \alpha(x,d) + \beta(x,d)v or u_{lin} = \alpha(y,d) + vif beta is set to None
        Args:
            nx (int): Number of states of your system (# of inputs for your controller).
            nu (int): Number of inputs of your system (# of outputs for your controller).
            nh (int): Number of hidden neurons per layer
            n_hidden_layers (int): Number of hidden layers
            beta (nn.Module, optional): Module that computes the beta decoupling matrix for feedback linearization.
                Defaults to None.
            nd (int): Number of measured disturbances the linearizing controller will absorb
                Defaults to 0
            lipschitz_bound (float): Upper bound on the Lipschitz constant for the neural controller parts
                Defaults to None
        """
    def __init__(self, nx: int, nu: int, nh: int, n_hidden_layers: int, beta: nn.Module = None, nd: int = 0, lipschitz_bound: float = None):
        super().__init__(nx + nd, nu, nh, n_hidden_layers, beta=beta, lipschitz_bound=lipschitz_bound)


    def forward(self, x: Tensor, u: Tensor, d:Tensor = None):
        """
            Args:
                x (Tensor): system's state 
                    Size : [batch_size, ny]
                u (Tensor): system's control input
                    Size : [batch_size, nu]
                d (Tensor) : measured disturbances 
                    Size : [batch_size, nd]
                    Defaults to None
        """
        if d is not None:
            inputs = torch.cat(x, d, dim=1)
        else:
            inputs = x
        
        return super().forward(inputs, u)