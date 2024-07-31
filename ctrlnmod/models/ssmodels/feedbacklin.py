import torch.nn as nn
import torch
from collections import OrderedDict
from .linear import NnLinear
from ...layers.layers import BetaLayer


class FLNSSM_decoupling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        output_dim: int,
        n_hid_layers: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, x]:
            z^+ = Az + B*beta(x)(u + alpha(x))
            y = Cz

        params :
            * input_dim : size of control input
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(FLNSSM_decoupling, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NnLinear(self.input_dim, self.state_dim, self.output_dim)
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Beta layer initialized to I_nu
        self.beta = BetaLayer(
            self.input_dim, self.state_dim, self.hidden_dim, self.actF
        )

        # Alpha layer

        self.alpha_in = nn.Linear(self.state_dim, self.hidden_dim)
        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(hidden_dim, hidden_dim))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.hidden_dim, self.state_dim, bias=False)

        # Nonlinear part initialized to 0
        nn.init.zeros_(self.alpha_out.weight)

    def forward(self, input, state):
        """
        params :
            - input = [u, x]
            - state =  state vector
        """
        u = input[:, 0: self.input_dim]
        x = input[:, self.input_dim: self.input_dim + self.state_dim]
        z = state

        alpha = self.alpha_in(x)
        alpha = self.actF(alpha)
        if self.n_hid_layers > 1:  # Only if there exists more than one hidden layer
            alpha = self.alpha_hid(alpha)
        alpha = self.alpha_out(alpha)

        beta = self.beta(x)

        input_lin = torch.bmm(beta, u + alpha)
        dz, y = self.linmod(input_lin, z)

        return dz, y

    def clone(self):
        copy = type(self)(
            self.input_dim,
            self.hidden_dim,
            self.state_dim,
            self.output_dim,
            self.n_hid_layers,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy


class QFLNSSM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        output_dim: int,
        n_hid_layers: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, x]:
            z^+ = Az + B*beta(x)(u + alpha(x)) + g(z,u)
            y = Cz

        params :
            * input_dim : size of control input
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(QFLNSSM, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NnLinear(self.input_dim, self.state_dim, self.output_dim)
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Beta layer by default it is initialized to the identity
        self.beta = BetaLayer(
            self.input_dim, self.state_dim, self.hidden_dim, self.actF
        )

        # Alpha layer

        self.alpha_in = nn.Linear(self.state_dim, self.hidden_dim)
        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(hidden_dim, hidden_dim))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.hidden_dim, self.input_dim, bias=False)

        # Nonlinear part initialized to 0
        nn.init.zeros_(self.alpha_out.weight)

        # g(z,u)
        self.g_in_z = nn.Linear(self.state_dim, self.hidden_dim)
        self.g_in_u = nn.Linear(self.input_dim, self.hidden_dim)
        if self.n_hid_layers > 1:
            paramsNLHidG = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(hidden_dim, hidden_dim))
                paramsNLHidG.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidG.append(tupact)
            self.g_hid = nn.Sequential(OrderedDict(paramsNLHidG))
        self.g_out = nn.Linear(self.hidden_dim, self.state_dim, bias=False)

    def forward(self, input, state):
        """
        params :
            - input = [u, x]
            - state =  state vector
        """
        u = input[:, 0: self.input_dim]
        x = input[:, self.input_dim: self.input_dim + self.state_dim]
        z = state

        alpha = self.alpha_in(x)
        alpha = self.actF(alpha)
        if self.n_hid_layers > 1:  # Only if there exists more than one hidden layer
            alpha = self.alpha_hid(alpha)
        alpha = self.alpha_out(alpha)

        beta = self.beta(x)

        input_lin = torch.bmm(beta, u + alpha)
        dz, y = self.linmod(input_lin, z)

        g = self.g_in_z(z) + self.g_in_u(u)
        g = self.actF(g)
        if self.n_hid_layers > 1:
            g = self.g_hid(g)
        g = self.g_out(g)
        dz = dz + g

        return dz, y

    def clone(self):
        copy = type(self)(
            self.input_dim,
            self.hidden_dim,
            self.state_dim,
            self.output_dim,
            self.n_hid_layers,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy


class FLNSSM_Elman(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        output_dim: int,
        n_hid_layers: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, x]:
            z^+ = Az + B(u + alpha(y))
            y = Cz

        params :
            * input_dim : size of control input
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(FLNSSM_Elman, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NnLinear(self.input_dim, self.state_dim, self.output_dim)
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Alpha layer

        self.alpha_in = nn.Linear(self.output_dim, self.hidden_dim)
        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(hidden_dim, hidden_dim))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.hidden_dim, self.input_dim)

        # Nonlinear part initialized to 0
        nn.init.zeros_(self.alpha_out.weight)

    def forward(self, input, state):
        """
        params :
            - input = [u, y]
            - state =  state vector
        """
        u = input[:, 0: self.input_dim]
        y = input[:, self.input_dim: self.input_dim + self.output_dim]
        z = state

        alpha = self.alpha_in()
        alpha = self.actF(alpha)
        if self.n_hid_layers > 1:  # Only if there exists more than one hidden layer
            alpha = self.alpha_hid(alpha)
        alpha = self.alpha_out(alpha)

        dz, y = self.linmod(u + alpha, z)

        return dz, y

    def clone(self):
        copy = type(self)(
            self.input_dim,
            self.hidden_dim,
            self.state_dim,
            self.output_dim,
            self.n_hid_layers,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy


class FLNSSM_Elman_Dist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        output_dim: int,
        n_hid_layers: int,
        nd: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, y, d]:
            z^+ = Az + B(u + alpha(y, d)) + Gd
            y = Cz

        params :
            * input_dim : size of control input
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(FLNSSM_Elman_Dist, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers
        self.dist_dim = nd

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NnLinear(
            self.input_dim + self.dist_dim, self.state_dim, self.output_dim
        )
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Alpha layer
        self.alpha_in_y = nn.Linear(
            self.output_dim, self.hidden_dim, bias=True)
        self.alpha_in_d = nn.Linear(self.dist_dim, self.hidden_dim, bias=False)

        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(hidden_dim, hidden_dim))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, input, state):
        """
        params :
            - input = [u, d]
            - state =  state vector
        """
        u = input[:, 0: self.input_dim]
        d = input[:, self.input_dim: self.input_dim + self.dist_dim]

        z = state
        y = self.linmod.C(z)

        alpha_y = self.alpha_in_y(y)
        alpha_d = self.alpha_in_d(d)
        alpha = self.actF(alpha_y + alpha_d)
        if self.n_hid_layers > 1:  # Only if there exists more than one hidden layer
            alpha = self.alpha_hid(alpha)
        alpha = self.alpha_out(alpha)

        dz, y = self.linmod(torch.cat((u + alpha, d), dim=1), z)

        return dz, y

    def init_weights(self, A0, B0, C0, isLinTrainable=True):
        # Initializing linear weights
        self.linmod.init_model_(A0, B0, C0, requires_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        nn.init.zeros_(self.alpha_out.weight)
        if self.alpha_out.bias is not None:
            nn.init.zeros_(self.alpha_out.bias)

        # TO-DO initialize inner weights to a specific distribution

    def load_weights(self, params_dict):
        """
        strWeigthsFile : MAT file containing wieghts matrices
        """
        # Linear part
        A0 = torch.Tensor(params_dict["A"])
        B0 = torch.Tensor(params_dict["B"])
        C0 = torch.Tensor(params_dict["C"])
        self.linmod.init_model_(A0, B0, C0, True)

        # Nonlinear part
        self.alpha_in_y.weight = nn.parameter.Parameter(
            torch.Tensor(params_dict["alpha_in_y"])
        )
        self.alpha_in_y.bias = nn.parameter.Parameter(
            torch.Tensor(params_dict["b_in"]).squeeze()
        )
        self.alpha_in_d.weight = nn.parameter.Parameter(
            torch.Tensor(params_dict["alpha_in_d"])
        )
        self.alpha_out.weight = nn.parameter.Parameter(
            torch.Tensor(params_dict["alpha_out"])
        )
        self.alpha_out.bias = nn.parameter.Parameter(
            torch.Tensor(params_dict["b_out"]).squeeze()
        )

    def clone(self):
        copy = type(self)(
            self.input_dim,
            self.hidden_dim,
            self.state_dim,
            self.output_dim,
            self.n_hid_layers,
            self.dist_dim,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy
