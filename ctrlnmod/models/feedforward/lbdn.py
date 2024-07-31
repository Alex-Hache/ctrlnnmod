'''
    Here are gathered all lipschitz bounded feedforward modules
    some are components of state-sapce models
'''
import torch
from torch.nn import Module, Linear, Sequential, Tanh
from collections import OrderedDict
from ctrl_nmod.layers.liplayers import SandwichLinear, SandwichLayer
from ctrl_nmod.linalg.utils import solveLipschitz


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

    def __init__(self, input_dim, hidden_dim, output_dim, actF=Tanh(), n_hidden=1) -> None:
        super(FFNN, self).__init__()
        self.nu = input_dim
        self.nh = hidden_dim
        self.ny = output_dim
        self.actF = actF
        self.n_hid = n_hidden

        layers = []

        # Input layer
        self.Win = Linear(self.nu, self.nh, bias=True)
        layers.append(('input_layer', self.Win))
        layers.append(('actF0', self.actF))

        # Hidden layers
        for k in range(self.n_hid - 1):  # If more than 1 hidden layer
            tup = ('dense{}'.format(k), Linear(
                hidden_dim, hidden_dim, bias=True))
            layers.append(tup)
            tupact = ('actF{}'.format(k), actF)
            layers.append(tupact)

        # Output layer
        self.Wout = Linear(self.nh, self.ny, bias=True)
        layers.append(('Out layer', self.Wout))

        self.fnn = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.fnn(x)


class Fxu(FFNN):
    r'''
        Create a feedforward neural network to be a nonlinear term in a state equation.

        .. ::math
            x^+ = Ax + Bu + f(x,u)
    '''

    def __init__(self, input_dim, hidden_dim, state_dim,
                 actF=Tanh(), n_hidden=1) -> None:
        super(Fxu, self).__init__(input_dim + state_dim, hidden_dim, state_dim,
                                  actF, n_hidden)
        self.nx = state_dim
        self.Wfx = self.Win.weight[:, :self.nx]
        self.Wfu = self.Win.weight[:, self.nx:]
        self.Wf = self.Wout.weight

    def forward(self, x, u):
        z = torch.cat((x, u), 1)
        return self.fnn(z)


class Hx(FFNN):
    '''
        Create a feedforward neural network h to be a nonlinear term in an output equation

        .. ::math
            y = Cx + h(x)
    '''

    def __init__(self, state_dim, hidden_dim, output_dim,
                 actF=Tanh(), n_hidden=1) -> None:
        super(Hx, self).__init__(state_dim, hidden_dim, output_dim,
                                 actF, n_hidden)
        self.nx = state_dim
        self.Wh = self.Wout.weight
        self.Whx = self.Win.weight

    def forward(self, x):
        return self.fnn(x)


class LBDN(Module):
    '''
        Create a Lipschitz Bounded Deep Neural network with a specific Lipschitz constant regarding every input channel.

        Taken from Pauli's LMI so only the input layer is scaled and not the output one
        compared to Wang.

        This can have effects on gradient propagation but allows
        for simple scaling of the Lipschitz constants of each individual input
        signal.

        See :     See https://github.com/acfr/LBDN for more details

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
                 act_f=Tanh(), n_hidden=1, param: str = 'expm') -> None:
        super(LBDN, self).__init__()

        self.nu = input_dim
        self.nh = hidden_dim
        self.ny = output_dim
        self.scale = scale
        self.act_f = act_f
        self.n_hid = n_hidden
        self.param = param

        layers = []
        self.Win = SandwichLayer(self.nu, self.nh,
                                 scale=self.scale, act_f=self.act_f, param=param)  # type: ignore
        layers.append(('input_layer', self.Win))
        for k in range(self.n_hid - 1):
            layer = SandwichLayer(self.nh, self.nh, act_f=act_f, param=param)  # type: ignore
            layers.append((f'hidden_layer_{k}', layer))
        self.Wout = SandwichLinear(self.nh, self.ny, param=param)
        layers.append(('output_layer', self.Wout))

        self.layers = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

    def check_(self):
        weights = self.extractWeightsSandwich()
        Wfx = weights[0][:, :self.nx]
        Wfu = weights[0][:, self.nx:]

        weights_x = [Wfx, weights[1]]
        weights_u = [Wfu, weights[1]]

        _, lx, _ = solveLipschitz(weights_x)
        _, lu, _ = solveLipschitz(weights_u)

        lip = torch.Tensor([lx] * self.nx + [lu] * self.nu)
        if torch.all(self.scale - lip > 0):
            return True
        else:
            return False

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
                    Q = layer.Q
                    At, B = Q[:, :f].T, Q[:, f:]
                    Win = (2**0.5) * torch.exp(-psi).diag() @ B * scle
                    Wout = (2**0.5) * At @ torch.exp(psi).diag()
                    weights_in.append(Win)
                    weights_out.append(Wout)  # weights after activation
                elif hasattr(layer, 'AB'):  # Sandwich lin class
                    fout, _ = layer.weight.shape
                    Q = layer.Q
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


class LipFxu(LBDN):
    def __init__(self, input_dim, hidden_dim, state_dim, scalex, scaleu,
                 act_f=Tanh(), n_hidden=1, param: str = 'expm') -> None:
        super().__init__(input_dim + state_dim, hidden_dim, state_dim,
                         torch.tensor([scalex] * state_dim + [scaleu] * input_dim), act_f, n_hidden, param=param)

        self.nx = state_dim
        self.Wfx = self.Win.weight[:, :self.nx]
        self.Wfu = self.Win.weight[:, self.nx:]
        self.Wf = self.Wout.weight

    def forward(self, x, u):
        z = torch.cat((x, u), 1)
        return self.layers(z)


class LipHx(LBDN):
    def __init__(self, state_dim, hidden_dim, output_dim, scalex,
                 actF=Tanh(), n_hidden=1, param: str = 'expm') -> None:
        super().__init__(state_dim, hidden_dim, output_dim,
                         torch.tensor([scalex] * state_dim), actF, n_hidden, param=param)
        self.Wh = self.Wout.weight
        self.Whx = self.Win.weight

    def forward(self, x):
        return self.layers(x)
