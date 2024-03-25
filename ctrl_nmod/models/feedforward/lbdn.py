'''
    Here are gathered all lipschitz bounded feedforward modules
    some are components of state-sapce models
'''
import torch
from torch.nn import Module, Linear, Sequential, Tanh
from collections import OrderedDict
from ctrl_nmod.layers.liplayers import SandwichFc, SandwichFcScaled, SandwichLin
from linalg.utils import solveLipschitz


class FFNN(Module):
    '''
        A class to define a feedforward neural network
    '''
    def __init__(self,  input_dim, hidden_dim, output_dim, actF=Tanh(), n_hidden=1) -> None:
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
            tup = ('dense{}'.format(k), Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(tup)
            tupact = ('actF{}'.format(k), actF)
            layers.append(tupact)

        # Output layer
        self.Wout = Linear(self.nh, self.ny, bias=True)
        layers.append(('Out layer', self.Wout))

        self.fnn = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.fnn(x)


class LBDN(Module):
    '''
        Create a Lipschitz Bounded Deep Neural network from Pauli's LMI
        so only the input layer is scaled and not the output one.*
        This can have effects on gradient propagation but allows
        for simple scaling of the Lipschitz constants of each individual input
        signal.
        See : LINK

        params:
            * input_dim : input dim
            * hidden_dim : hidden layer dim (they all have the same)
            * output_dim : output dim
            * scale : the scaling (vector) to be used to scale the input
            * actF : the activation function used
            * n_hidden : number of hidden layers

    '''
    def __init__(self, input_dim, hidden_dim, output_dim, scale,
                 actF=Tanh(), n_hidden=1) -> None:
        super(LBDN, self).__init__()

        self.nu = input_dim
        self.nh = hidden_dim
        self.ny = output_dim
        self.scale = scale
        self.actF = actF
        self.n_hid = n_hidden

        layers = []
        self.Win = SandwichFcScaled(self.nu, self.nh,
                                    scale=self.scale, actF=self.actF)  # type: ignore
        layers.append(('input_layer', self.Win))
        for k in range(self.n_hid-1):
            layer = SandwichFc(self.nh, self.nh, actF=actF)  # type: ignore
            layers.append((f'hidden_layer_{k}', layer))
        self.Wout = SandwichLin(self.nh, self.ny)
        layers.append(('output_layer', self.Wout))

        self.layers = Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)


class Fxu(FFNN):
    '''
        Create a feedforward neural network to be a nonlinear term in a state equation
        x+ = Ax + Bu + f(x,u)
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

    def check_(self):
        weights = self.extractWeightsSandwich()
        Wfx = weights[0][:, :self.nx]
        Wfu = weights[0][:, self.nx:]

        weights_x = [Wfx, weights[1]]
        weights_u = [Wfu, weights[1]]

        _, lx, _ = solveLipschitz(weights_x)
        _, lu, _ = solveLipschitz(weights_u)

        lip = torch.Tensor([lx]*self.nx + [lu]*self.nu)
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
                        weights_out.append(2*Q[:, :fout].T)  # Weights after activation
                else:  # regular weights
                    weights_in.append(layer.weight)

        # Combine sandwich layers
        weights = []
        weights.append(weights_in[0])  # W0

        for k in range(len(weights_in)-1):
            weights.append(weights_in[k+1] @ weights_out[k])

        return weights


class Hx(FFNN):
    '''
        Create a feedforward neural network h to be a nonlinear term in an output equation
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


class LipFxu(LBDN):
    def __init__(self, input_dim, hidden_dim, state_dim, scalex, scaleu,
                 actF=Tanh(), n_hidden=1) -> None:
        super().__init__(input_dim + state_dim, hidden_dim, state_dim,
                         torch.tensor([scalex]*state_dim + [scaleu]*input_dim), actF, n_hidden)

        self.nx = state_dim
        self.Wfx = self.Win.weight[:, :self.nx]
        self.Wfu = self.Win.weight[:, self.nx:]
        self.Wf = self.Wout.weight

    def forward(self, x, u):
        z = torch.cat((x, u), 1)
        return self.layers(z)


class LipHx(LBDN):
    def __init__(self, state_dim, hidden_dim, output_dim, scalex,
                 actF=Tanh(), n_hidden=1) -> None:
        super().__init__(state_dim, hidden_dim, output_dim,
                         torch.tensor([scalex]*state_dim), actF, n_hidden)
        self.Wh = self.Wout.weight
        self.Whx = self.Win.weight

    def forward(self, x):
        return self.layers(x)
