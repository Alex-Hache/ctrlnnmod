'''
    Here are gathered all lipschitz bounded feedforward modules
    some are components of state-sapce models
'''
import torch
from torch.nn import Module, Linear, Sequential, Tanh
from collections import OrderedDict
from ctrlnmod.layers.liplayers import SandwichLinear, SandwichLayer
from ctrlnmod.lmis.lipschitz import LipschitzLMI
from typing import Optional


class FFNN(Module):
    '''
        A class to define a feedforward neural network

    attributes
    ----------

        * nu : int
            input dimension
        * nh : int
            hidden_dimension
        * ny : int
            output dimension
        * actF : (default Tanh)
            activation function
        * n_hidden : int
            number of hidden layers
        * fnn : nn.Sequential
            a simple sequential feedforward model
    '''

    def __init__(self, input_dim, hidden_dim, output_dim,
                 act_f: Optional[Module] = Tanh(), n_hidden=1, bias=True) -> None:
        super(FFNN, self).__init__()
        self.nu = input_dim
        self.nh = hidden_dim
        self.ny = output_dim
        self.act_f = act_f
        self.n_hid = n_hidden
        self.bias = bias

        layers = []

        # Input layer
        self.Win = Linear(self.nu, self.nh, bias=bias)
        layers.append(('input_layer', self.Win))
        layers.append(('actF0', self.act_f))

        # Hidden layers
        for k in range(self.n_hid - 1):  # If more than 1 hidden layer
            tup = ('dense{}'.format(k), Linear(
                hidden_dim, hidden_dim, bias=self.bias))
            layers.append(tup)
            tupact = ('actF{}'.format(k + 1), act_f)
            layers.append(tupact)

        # Output layer  -- no bias
        self.Wout = Linear(self.nh, self.ny, bias=True)
        layers.append(('Out layer', self.Wout))

        self.layers = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

    def init_weights_(self):
        torch.nn.init.zeros_(self.Wout.weight)  # Initializing the output layer weights to 0

    def get_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                weights.append(layer.weight)
        return weights
    

class Fxu(FFNN):
    r'''
        Create a feedforward neural network to be a nonlinear term in a state equation.

        .. ::math
            x^+ = Ax + Bu + f(x,u)
    '''

    def __init__(self, input_dim, hidden_dim, state_dim,
                 actF: Module = Tanh(), n_hidden=1, bias=True) -> None:
        super(Fxu, self).__init__(input_dim + state_dim, hidden_dim, state_dim,
                                  actF, n_hidden, bias=bias)
        self.nx = state_dim
        self.Wfx = self.Win.weight[:, :self.nx]
        self.Wfu = self.Win.weight[:, self.nx:]
        self.Wf = self.Wout.weight

    def forward(self, x, u):
        z = torch.cat((x, u), 1)
        return self.layers(z)


class Hx(FFNN):
    '''
        Create a feedforward neural network h to be a nonlinear term in an output equation

        .. ::math
            y = Cx + h(x)
    '''

    def __init__(self, state_dim, hidden_dim, output_dim,
                 actF: Module = Tanh(), n_hidden=1, bias=True) -> None:
        super(Hx, self).__init__(state_dim, hidden_dim, output_dim,
                                 actF, n_hidden, bias=bias)
        self.nx = state_dim
        self.Wh = self.Wout.weight
        self.Whx = self.Win.weight

    def forward(self, x):
        return self.layers(x)


class LBDN(Module):
    '''
        Create a Lipschitz Bounded Deep Neural network with a specific Lipschitz constant regarding every input channel.

        Taken from Pauli's LMI so only the input layer is scaled and not the output one
        compared to Wang.

        This can have effects on gradient propagation but allows
        for simple scaling of the Lipschitz constants of each individual input
        signal.

        See : https://github.com/acfr/LBDN for more details

        attributes
        ------
            * input_dim : int
                input dimension
            * hidden_dim : int
                hidden layer dim (they all have the same)
            * output_dim : int
                output dim
            * scale : Tensor
                the scaling to be used to scale the input in the Lipschitez layers
                Must be of size nu + nx
            * act_f : nn.Module
                the activation function used
            * n_hidden : int
                number of hidden layers

    '''

    def __init__(self, input_dim, hidden_dim, output_dim, scale,
                 act_f: Module = Tanh(), n_hidden=1, param: str = 'expm', bias=True) -> None:
        super(LBDN, self).__init__()

        self.nu = input_dim
        self.nh = hidden_dim
        self.ny = output_dim
        self.scale = scale
        self.act_f = act_f
        self.n_hid = n_hidden
        self.param = param
        self.bias = bias

        layers = []
        self.Win = SandwichLayer(self.nu, self.nh,
                                 scale=self.scale, act_f=self.act_f, param=param,
                                 bias=bias)  # type: ignore
        layers.append(('input_layer', self.Win))
        for k in range(self.n_hid - 1):
            layer = SandwichLayer(
                self.nh, self.nh, act_f=act_f, param=param, bias=bias)  # type: ignore
            layers.append((f'hidden_layer_{k}', layer))
        # No bias on the output layer
        self.Wout = SandwichLinear(self.nh, self.ny, param=param, bias=False)
        layers.append(('output_layer', self.Wout))

        self.layers = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

    def extractWeightsSandwich(self):
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
        weights = self.extractWeightsSandwich()

        _, l, _ = LipschitzLMI.solve(weights)

        lip = torch.Tensor([l] * self.nu)

        infos = {'lip': round(l.item(), 3)}
        if torch.all(self.scale - lip > 0):
            return True, infos
        else:
            return False, infos
    def init_weights_(self):
        pass
class LipFxu(LBDN):
    def __init__(self, input_dim, hidden_dim, state_dim, scalex, scaleu,
                 act_f=Tanh(), n_hidden=1, param: str = 'expm', bias=True) -> None:
        super().__init__(input_dim + state_dim, hidden_dim, state_dim,
                         torch.tensor([scalex] * state_dim + [scaleu] * input_dim), act_f, n_hidden, param=param, bias=bias)
        self.nu = input_dim
        self.nx = state_dim
        self.Wfx = self.Win.weight[:, :self.nx]
        self.Wfu = self.Win.weight[:, self.nx:]
        self.Wf = self.Wout.weight

    def forward(self, x, u):
        z = torch.cat((x, u), 1)
        return self.layers(z)

    def check(self):
        weights = self.extractWeightsSandwich()
        Wfx = weights[0][:, :self.nx]
        Wfu = weights[0][:, self.nx:]

        weights_x = [Wfx, *weights[1:]]
        weights_u = [Wfu, *weights[1:]]

        _, lx, _ = LipschitzLMI.solve(weights_x)
        _, lu, _ = LipschitzLMI.solve(weights_u)

        lip = torch.Tensor([lx] * self.nx + [lu] * self.nu)

        infos = {'lipx': round(lx.item(), 3), 'lipu': round(lu.item(), 3)}
        if torch.all(self.scale - lip > 0):
            return True, infos
        else:
            return False, infos

class LipHx(LBDN):
    def __init__(self, state_dim, hidden_dim, output_dim, scalex,
                 actF=Tanh(), n_hidden=1, param: str = 'expm', bias=True) -> None:
        super().__init__(state_dim, hidden_dim, output_dim,
                         torch.tensor([scalex] * state_dim), actF, n_hidden, param=param, bias=bias)
        self.Wh = self.Wout.weight
        self.Whx = self.Win.weight

    def forward(self, x):
        return self.layers(x)
