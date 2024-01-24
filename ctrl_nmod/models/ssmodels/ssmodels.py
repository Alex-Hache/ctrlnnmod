
from torch.nn import Module, Tanh, Linear
from torch.nn.parameter import Parameter
from models.feedforward.lbdn import Fxu, LipFxu, LipHx, Hx
from torch.nn.init import zeros_
from typing import Tuple
from linalg.matrices import AlphaStable


class NnLinear(Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim: int, output_dim: int, state_dim: int,
                 alpha=None) -> None:
        super(NnLinear, self).__init__()

        self.nu = input_dim
        self.nx = state_dim
        self.ny = output_dim

        # Is A alpha stable ?
        if alpha is not None:
            self.A = AlphaStable(self.nx, alpha=alpha)
        else:
            self.A = Linear(self.nx, self.nx, bias=False)
        self.B = Linear(self.nu, self.nu, bias=False)
        self.C = Linear(self.nx, self.ny, bias=False)

    def forward(self, u, x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.A.weight = Parameter(A0)
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy


class Grnssm(Module):
    def __init__(self, input_dim: int, output_dim: int, state_dim: int,
                 hidden_dim: int, n_hidden_layers: int = 1, actF=Tanh(),
                 out_eq_nl=False, alpha=None) -> None:
        """
        Constructor Grnssm u is a generalized input ex : [control, distrubance]:
            x^+ = Ax + Bu + f(x,u)
            y = Cx + h(x)

        params :
            * input_dim : size of input layer
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
            * out_eq : nonlinear output equation
            * alpha : alpha-stability bound for A matrix
        """
        super(Grnssm, self).__init__()

        # Set network dimensions
        self.nu = input_dim
        self.nx = state_dim
        self.nh = hidden_dim
        self.ny = output_dim
        self.n_hid_layers = n_hidden_layers

        # Activation functions
        self.actF = actF

        # Nonlinear output equation
        self.out_eq_nl = out_eq_nl

        # Linear part
        self.linmod = NnLinear(self.nu, self.ny, self.nx, alpha=alpha)

        # Nonlinear state part
        self.fx = Fxu(self.nu, self.nh, self.nx)
        if self.out_eq_nl:
            self.hx = Hx(self.nx, self.nh, self.ny)

    def forward(self, u, x):
        # Forward pass -- prediction of the output at time k : y_k
        x_lin, y_lin = self.linmod(u, x)  # Linear part

        # Nonlinear part fx
        fx = self.fx(x, u)
        dx = x_lin + fx

        if self.out_eq_nl:
            hx = self.hx(x)
            y = y_lin + hx
        else:
            y = y_lin
        return dx, y

    def init_weights_(self, A0, B0, C0, isLinTrainable=True) -> None:
        # TODO Enforce specific distribution ton inner (and outer) weights
        # Initializing linear weights
        self.linmod.init_weights_(A0, B0, C0, is_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        zeros_(self.fx.Wout.weight)
        if self.fx.Wout.bias is not None:
            zeros_(self.fx.Wout.bias)
        # zeros_(self.Wh.weight)
        if self.out_eq_nl:
            zeros_(self.hx.Wout.weight)
            if self.hx.Wout.bias is not None:
                zeros_(self.hx.Wout.bias)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx,
                          self.nh, self.n_hid_layers, self.actF,
                          self.out_eq_nl)
        copy.load_state_dict(self.state_dict())
        return copy


class LipGrnssm(Grnssm):
    def __init__(self, input_dim: int, output_dim: int, state_dim: int,
                 hidden_dim: int, n_hidden_layers: int = 1, actF=Tanh(),
                 out_eq_nl=False, lip: Tuple[int, int] = (1, 1)) -> None:
        super(LipGrnssm, self).__init__(input_dim, output_dim, state_dim,
                                        hidden_dim, n_hidden_layers,
                                        actF, out_eq_nl)

        # We override the definition of the nonlinear parts of the network
        # and replace them with LBDN networks

        # Nonlinear state part
        self.lip_x = lip[0]
        self.lip_u = lip[1]

        self.fx = LipFxu(self.nu, self.nh, self.nx, scalex=self.lip_x, scaleu=self.lip_u)
        # Nonlinear output part
        if self.out_eq_nl:
            self.hx = LipHx(self.nx, self.nh, self.ny, scalex=self.lip_x)

    def forward(self, u, x):
        return super().forward(u, x)

    def init_weights_(self, A0, B0, C0, isLinTrainable=True):
        super().init_weights_(A0, B0, C0, isLinTrainable)
