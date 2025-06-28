'''
    Here are gathered all lipschitz bounded feedforward modules
    some are components of state-sapce models
'''
import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, Tanh
from collections import OrderedDict
from ctrlnmod.layers.liplayers import SandwichLinear, SandwichLayer
from ctrlnmod.lmis.lipschitz import Lipschitz
from typing import Optional, List, Union
from ctrlnmod.utils import parse_act_f
from ctrlnmod.linalg.utils import fill_strictly_block_triangular


class FFNN(Module):
    r"""
    A fully connected feedforward neural network (MLP) with configurable depth and width.

    The architecture is defined by a list of hidden layer sizes. The model consists of
    linear layers followed by activation functions, with the final output layer having no bias.

    Args:
        n_in (int): Dimension of the input features.
        hidden_layers (List[int]): List specifying the width of each hidden layer.
        n_out (int): Dimension of the output features.
        act_f (Optional[nn.Module]): Activation function applied between layers. Default is Tanh().
        bias (bool): Whether to include bias terms in linear layers. Default is True.

    Attributes:
        n_in (int): Dimension of the input.
        hidden_layers (List[int]): Widths of the hidden layers.
        n_out (int): Dimension of the output.
        act_f (nn.Module): Activation function used between layers.
        bias (bool): Whether linear layers include bias.
        layers (nn.Sequential): Complete feedforward model.
    """

    def __init__(
        self,
        n_in: int,
        hidden_layers: List[int],
        n_out: int,
        act_f: str = 'relu',
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.n_in = n_in
        self.hidden_layers = hidden_layers
        self.n_out = n_out
        self.act_f =  parse_act_f(act_f)
        self.act_f_str = act_f
        self.bias = bias

        layers = OrderedDict()

        # Parse activation function

        # Input layer
        layers["input"] = Linear(n_in, hidden_layers[0], bias=bias)
        layers["act_0"] = self.act_f

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers[f"linear_{i}"] = Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias)
            layers[f"act_{i}"] = self.act_f

        # Output layer (no bias)
        layers["output"] = Linear(hidden_layers[-1], n_out, bias=False)

        self.layers = Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the output of the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(..., n_in)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(..., n_out)``.
        """
        return self.layers(x)

    def get_weights(self) -> List[torch.Tensor]:
        r"""
        Returns the weight matrices of all linear layers in the network.

        Returns:
            List[torch.Tensor]: A list containing the weight tensors.
        """
        weights = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                weights.append(layer.weight)
        return weights

    def clone(self) -> 'FFNN':
        r"""
        Creates a deep copy of the current feedforward network.

        Returns:
            FFNN: A new instance of FFNN with the same architecture and weights.
        """
        copy = FFNN(
            n_in=self.n_in,
            hidden_layers=self.hidden_layers,
            n_out=self.n_out,
            act_f=self.act_f_str,
            bias=self.bias,
        )
        copy.load_state_dict(self.state_dict())
        return copy
    
    def to_snof(self):
        # Extract weights as a list
        weights = self.get_weights()

        # Compute sizes
        nh_1, n_in = weights[0].shape
        n_out, nh_l = weights[-1].shape
        nz = sum([weight.shape[0] for weight in weights[:-1]])

        A = torch.zeros((nz, nz))
        B = torch.zeros((nz, n_in))
        C = torch.zeros((n_out, nz))

        B[:nh_1, :] = weights[0]
        C[:, -nh_l:] = weights[-1]

        weights_inter = weights[1:-1]
        if weights_inter:  # If there are more then 1 hidden layer then this should be not empty
            A = fill_strictly_block_triangular(A, weights_inter, 'lower')
        return A, B, C


    def init_weights_(self, init=torch.nn.init.kaiming_uniform_) -> None:
            r"""
            Initializes the weights of all linear layers using the specified initialization method
            and activation-aware gain. Biases (if present) are initialized to zero.

            Args:
                init (Callable): Initialization function from `torch.nn.init`. Default is `kaiming_uniform_`.
            """
            gain = torch.nn.init.calculate_gain(self.act_f_str)

            for layer in self.layers:
                if isinstance(layer, Linear):
                    init(layer.weight, a=gain)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

class LBDN(Module):
    r"""
    Lipschitz Bounded Deep Neural Network (LBDN) with input-wise scaling.

    This implementation applies a structured initialization that bounds the 
    Lipschitz constant of the entire network based on a per-input scaling vector.
    Only the input layer is explicitly scaled, following the construction proposed 
    in Wang and Manchester (see reference below), which simplifies the constraint structure.

    Reference:
        https://github.com/acfr/LBDN

    Args:
        n_in (int): Dimension of the input vector.
        hidden_dim (int): Width of each hidden layer.
        n_out (int): Dimension of the output vector.
        scale (float | Tensor): Input-wise scaling tensor of shape ``(n_in,)``.
        act_f (str): Activation function to apply between layers. Default: ``relu``.
        n_hidden (int): Number of hidden layers. Default: ``1``.
        param (str): Parameterization mode for Lipschitz constraints (e.g., ``'expm'``). Default: ``'expm'``.
        bias (bool): Whether to include bias terms in the layers. Default: ``True``.

    Attributes:
        layers (nn.Sequential): The complete feedforward model with sandwich layers.
    """

    def __init__(self, n_in: int, hidden_layers: List[int], n_out: int, scale: Tensor,
                 act_f: str ='relu', param: str = 'expm', bias=True) -> None:
        super(LBDN, self).__init__()

        self.nu = n_in
        self.hidden_layers = hidden_layers
        self.ny = n_out
        self.scale = scale
        self.act_f = parse_act_f(act_f)
        self.act_f_str = act_f
        self.n_hid = len(hidden_layers)
        self.param = param
        self.bias = bias

        layers = []
        self.Win = SandwichLayer(self.nu, self.hidden_layers[0],
                                 scale=self.scale, act_f=self.act_f, param=param,
                                 bias=bias)  # type: ignore
        layers.append(('input', self.Win))
        for k in range(self.n_hid - 1):
            layer = SandwichLayer(
                self.hidden_layers[k], self.hidden_layers[k+1], act_f=self.act_f, param=param, bias=bias)  # type: ignore
            layers.append((f'hidden_layer_{k}', layer))
        # No bias on the output layer
        self.Wout = SandwichLinear(self.hidden_layers[-1], self.ny, param=param, bias=False)
        layers.append(('output', self.Wout))

        self.layers = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

    def get_weights(self) -> List[Tensor]:
        '''
            Model must something similar to a sequential model
            with an attribute layers being an iterable containing the weights
        '''
        weights_in = []
        weights_out = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                if hasattr(layer, 'scale'):
                    scle = layer.scale
                else:
                    scle = 1
                if hasattr(layer, 'psi'):  # Sandwich fc class
                    psi = layer.psi
                    f = psi.shape[0]
                    Q = layer.weight
                    At, B = Q[:, :f].T, Q[:, f:]
                    Win = (2**0.5) * torch.exp(-psi).diag() @ B * scle
                    Wout = (2**0.5) * At @ torch.exp(psi).diag()
                    weights_in.append(Win)
                    weights_out.append(Wout)  # weights after activation
                elif hasattr(layer, 'AB'):  # Sandwich lin class
                    fout, _ = layer.weight.shape
                    Q = layer.weight
                    weights_in.append(Q[:, fout:] * scle)
                    if layer.AB:
                        # Weights after activation
                        weights_out.append(2 * Q[:, :fout].T)
                else:  # regular weights
                    weights_in.append(layer.weight)

        # Combine sandwich layers
        weights = []
        weights.append(weights_in[0])  # W0

        for k in range(len(weights_in) - 1):
            weights.append(weights_in[k + 1] @ weights_out[k])

        return weights

    def check(self):
        A, B, C = self.to_snof()
        _, l, _ = Lipschitz.solve(A, B, C)

        lip = torch.Tensor([l] * self.nu)

        infos = {'lip': round(l.item(), 3)}
        if torch.all(self.scale - lip > 0):
            return True, infos
        else:
            return False, infos

    def clone(self) -> 'LBDN':
        """
        Creates a deep copy of the current LBDN model.

        Returns:
            LBDN: A new instance of LBDN with the same architecture and weights.
        """
        copy = LBDN(
            n_in=self.nu,
            hidden_layers=self.hidden_layers,
            n_out=self.ny,
            scale=self.scale,
            act_f=self.act_f_str,
            param=self.param,
            bias=self.bias
        )
        copy.load_state_dict(self.state_dict())
        return copy
    
    def init_weights_(self, init=torch.nn.init.kaiming_uniform_):
        """
            For now we stick to the default initialization from wang & Manchester
            See ctrlnmod.layers.liplayers for info
        """
        pass 


    def to_snof(self, weights: Optional[List[Tensor]]=None):
        # Extract weights as a list
        if weights is None:
            weights = self.get_weights()

        # Compute sizes
        nh_1, n_in = weights[0].shape
        n_out, nh_l = weights[-1].shape
        nz = sum([weight.shape[0] for weight in weights[:-1]])

        A = torch.zeros((nz, nz))
        B = torch.zeros((nz, n_in))
        C = torch.zeros((n_out, nz))

        B[:nh_1, :] = weights[0]
        C[:, -nh_l:] = weights[-1]

        weights_inter = weights[1:-1]
        if weights_inter:  # If there are more then 1 hidden layer then this should be not empty
            A = fill_strictly_block_triangular(A, weights_inter, 'lower')
        return A, B, C
    

class LipFxu(LBDN):
    def __init__(self, n_in, hidden_dim, state_dim, scalex, scaleu,
                 act_f=Tanh(), n_hidden=1, param: str = 'expm', bias=True) -> None:
        super().__init__(n_in + state_dim, hidden_dim, state_dim,
                         torch.tensor([scalex] * state_dim + [scaleu] * n_in), act_f, n_hidden, param=param, bias=bias)
        self.nu = n_in
        self.nx = state_dim
        self.Wfx = self.Win.weight[:, :self.nx]
        self.Wfu = self.Win.weight[:, self.nx:]
        self.Wf = self.Wout.weight

    def forward(self, x, u):
        z = torch.cat((x, u), 1)
        return self.layers(z)

    def check(self):
        weights = self.get_weights()
        Wfx = weights[0][:, :self.nx]
        Wfu = weights[0][:, self.nx:]

        weights_x = [Wfx, *weights[1:]]
        weights_u = [Wfu, *weights[1:]]

        
        _, lx, _ = Lipschitz.solve(*self.to_snof(weights_x))
        _, lu, _ = Lipschitz.solve(*self.to_snof(weights_u))

        lip = torch.Tensor([lx] * self.nx + [lu] * self.nu)

        infos = {'lipx': round(lx.item(), 3), 'lipu': round(lu.item(), 3)}
        if torch.all(self.scale - lip > 0):
            return True, infos
        else:
            return False, infos