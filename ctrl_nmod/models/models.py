import torch.nn as nn
import torch
from nssmid.linalg_utils import *
from collections import OrderedDict
from nssmid.layers import *


class NNLinear(nn.Module):
    """
    neural network module corresponding to a linear state-space model
        x^+ = Ax + Bu
        y = Cx
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int) -> None:
        super(NNLinear, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.A = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.B = nn.Linear(self.input_dim, self.state_dim, bias=False)
        self.C = nn.Linear(self.state_dim, self.output_dim, bias=False)

    def forward(self, u, x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx, y

    def init_model_(self, A0, B0, C0, is_grad=True):
        self.A.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)


class GRNSSM(nn.Module):
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
        Constructor grNSSM u is a generalized input ex : [control, distrubance]:
            x^+ = Ax + Bu + f(x,u)
            y = Cx + h(x)

        params :
            * input_dim : size of input layer
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(GRNSSM, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers
        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NNLinear(self.input_dim, self.state_dim, self.output_dim)
        # Nonlinear part for the state f(x,u) = W_l \sigma(W_{l-1}...)
        # Input Layer
        self.Wfx = nn.Linear(self.state_dim, self.hidden_dim, bias=False)
        self.Wfu = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.actFinF = actF
        if self.n_hid_layers > 1:
            paramsNLHidF = []
            for k in range(n_hid_layers - 1):
                tup = (
                    "dense{}".format(k),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                )
                paramsNLHidF.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidF.append(tupact)
            self.f_hid = nn.Sequential(OrderedDict(paramsNLHidF))
        self.Wf = nn.Linear(self.hidden_dim, self.state_dim, bias=True)

        """
        # Nonlinear part for the output y = cx + h(x)
        self.Whx = nn.Linear(self.state_dim, self.hidden_dim, bias = False)
        self.actFinH = actF
        if self.n_hid_layers>1:
            paramsNLHidH = []
            for k in range(n_hid_layers-1):
                tup = ('dense{}'.format(k), nn.Linear(hidden_dim, hidden_dim, bias= False))
                paramsNLHidH.append(tup)
                tupact = ('actF{}'.format(k), actF)
                paramsNLHidH.append(tupact)
            self.h_hid = nn.Sequential(OrderedDict(paramsNLHidH))
        self.Wh = nn.Linear(self.hidden_dim, self.output_dim, bias = False)
        """

    def forward(self, u, x):
        # Forward pass -- prediction of the output at time k : y_k

        x_lin, y_lin = self.linmod(u, x)  # Linear part

        # Nonlinear part fx
        fx = self.Wfx(x) + self.Wfu(u)
        fx = self.actFinF(fx)
        if self.n_hid_layers > 1:
            fx = self.f_hid(fx)
        fx = self.Wf(fx)

        dx = x_lin + fx

        """
        # Nonlinear part hx
        hx = self.Whx(x)
        hx = self.actFinH(hx)

        if self.n_hid_layers>1:
            hx = self.h_hid(hx)
        hx = self.Wh(hx)
        """
        y = y_lin  # + hx

        return dx, y

    def init_weights(self, A0, B0, C0, isLinTrainable=True):
        # Initializing linear weights
        self.linmod.init_model_(A0, B0, C0, is_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        nn.init.zeros_(self.Wf.weight)
        if self.Wf.bias is not None:
            nn.init.zeros_(self.Wf.bias)
        # nn.init.zeros_(self.Wh.weight)

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
        self.Wfu.weight = nn.parameter.Parameter(torch.Tensor(params_dict["Wfu"]))
        self.Wfu.bias = nn.parameter.Parameter(
            torch.Tensor(params_dict["bi"]).squeeze()
        )
        self.Wfx.weight = nn.parameter.Parameter(torch.Tensor(params_dict["Wfx"]))
        self.Wf.weight = nn.parameter.Parameter(torch.Tensor(params_dict["Wf"]))
        self.Wf.bias = nn.parameter.Parameter(torch.Tensor(params_dict["bo"]).squeeze())

    def clone(self):  # Method called by the simulator
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


class GRNSSM_dist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        output_dim: int,
        n_hid_layers: int,
        dist_dim: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor grNSSM u is a generalized input ex : [control, distrubance]:
            x^+ = Ax + Bu + f(x,u)
            y = Cx + h(x)

        params :
            * input_dim : size of input layer
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(GRNSSM_dist, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers
        self.dist_dim = dist_dim

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NNLinear(
            self.input_dim + self.dist_dim, self.state_dim, self.output_dim
        )
        # Nonlinear part for the state f(x,u) = W_l \sigma(W_{l-1}...)
        # Input Layer
        self.Wfx = nn.Linear(self.state_dim, self.hidden_dim, bias=True)
        self.Wfu = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.Wfd = nn.Linear(self.dist_dim, self.hidden_dim, bias=False)
        self.actFinF = actF
        if self.n_hid_layers > 1:
            paramsNLHidF = []
            for k in range(n_hid_layers - 1):
                tup = (
                    "dense{}".format(k),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                )
                paramsNLHidF.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidF.append(tupact)
            self.f_hid = nn.Sequential(OrderedDict(paramsNLHidF))
        self.Wf = nn.Linear(self.hidden_dim, self.state_dim, bias=True)

        """
        # Nonlinear part for the output y = cx + h(x)
        self.Whx = nn.Linear(self.state_dim, self.hidden_dim, bias = True)
        self.actFinH = actF
        if self.n_hid_layers>1:
            paramsNLHidH = []
            for k in range(n_hid_layers-1):
                tup = ('dense{}'.format(k), nn.Linear(hidden_dim, hidden_dim, bias= False))
                paramsNLHidH.append(tup)
                tupact = ('actF{}'.format(k), actF)
                paramsNLHidH.append(tupact)
            self.h_hid = nn.Sequential(OrderedDict(paramsNLHidH))
        self.Wh = nn.Linear(self.hidden_dim, self.output_dim, bias = False)
        """

    def forward(self, input, x):
        # Forward pass -- prediction of the output at time k : y_k

        u = input[:, 0 : self.input_dim]
        d = input[:, self.input_dim : self.input_dim + self.dist_dim]
        x_lin, y_lin = self.linmod(input, x)  # Linear part

        # Nonlinear part fx
        fx = self.Wfx(x) + self.Wfu(u) + self.Wfd(d)
        fx = self.actFinF(fx)
        if self.n_hid_layers > 1:
            fx = self.f_hid(fx)
        fx = self.Wf(fx)

        dx = x_lin + fx

        """
    
        # Nonlinear part hx
        hx = self.Whx(x)
        hx = self.actFinH(hx)

        if self.n_hid_layers>1:
            hx = self.h_hid(hx)
        hx = self.Wh(hx)
        """
        y = y_lin  # + hx

        return dx, y

    def init_weights(self, A0, B0, C0, isLinTrainable=True):
        # Initializing linear weights
        self.linmod.init_model_(A0, B0, C0, is_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        nn.init.zeros_(self.Wf.weight)
        # nn.init.zeros_(self.Wh.weight)

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

    def clone(self):  # Method called by the simulator
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
        self.linmod = NNLinear(self.input_dim, self.state_dim, self.output_dim)
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
        u = input[:, 0 : self.input_dim]
        x = input[:, self.input_dim : self.input_dim + self.state_dim]
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
        self.linmod = NNLinear(self.input_dim, self.state_dim, self.output_dim)
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
        u = input[:, 0 : self.input_dim]
        x = input[:, self.input_dim : self.input_dim + self.state_dim]
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
        self.linmod = NNLinear(self.input_dim, self.state_dim, self.output_dim)
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
        u = input[:, 0 : self.input_dim]
        y = input[:, self.input_dim : self.input_dim + self.output_dim]
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
        self.linmod = NNLinear(
            self.input_dim + self.dist_dim, self.state_dim, self.output_dim
        )
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Alpha layer
        self.alpha_in_y = nn.Linear(self.output_dim, self.hidden_dim, bias=True)
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
        u = input[:, 0 : self.input_dim]
        d = input[:, self.input_dim : self.input_dim + self.dist_dim]

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
        self.linmod.init_model_(A0, B0, C0, is_grad=isLinTrainable)

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


class PINN(nn.Module):
    def __init__(self, nq, nx, nh, nu, ny, actF=nn.Tanh()) -> None:
        super(PINN, self).__init__()

        self.nq = nq
        self.nh = nh
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # Intertia Matrix
        self.m = InertiaMatrix(nq, nh)  # m(x_1, x_2)

        # Coriolis term c(x) = c(x_1, x_2, x_3, x_4)
        self.c_in = nn.Linear(nx, nh, bias=False)
        self.actF = actF
        self.c_out = nn.Linear(nh, nq, bias=False)

        # Gravity term
        self.g_in = nn.Linear(nq, nh, bias=False)
        self.g_out = nn.Linear(nh, nq, bias=False)

    def forward(self, u, x):

        q = x[:, :2]
        q_dot = x[:, 2:]
        cx = self.c_in(x)
        cx = self.actF(cx)
        cx = self.c_out(cx)

        gx = self.g_in(q)
        gx = self.actF(gx)
        gx = self.g_out(gx)

        m = self.m(q)

        dqq = torch.bmm(m, -cx - gx + u)
        dx = torch.cat((q_dot, dqq), dim=1)
        y = q
        return dx, y


class DoublePendulum(nn.Module):
    def __init__(self, l1=1, l2=0.9, m1=5, m2=4.9, f=2) -> None:
        super(DoublePendulum, self).__init__()

        # Cinematic parameters
        self.l1 = l1
        self.l2 = l2
        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2
        self.m1 = m1
        self.m2 = m2
        self.f = f
        self.g = 9.81

        # Geometric parameters
        self.a1 = self.m1 * self.d1**2 + self.m2 * l1**2
        self.a2 = self.m2 * self.d2**2
        self.a3 = self.m2 * self.l1 * self.d2
        self.a4 = self.m1 * self.d1 + self.m2 * self.l1
        self.a5 = self.m2 * self.d2

    def forward(self, u, x):
        batch_size = x.shape[0]
        q = x[:, :2]
        q1 = x[:, 0]
        q2 = x[:, 1]
        q_dot = x[:, 2:]
        q1_dot = x[:, 2]
        q2_dot = x[:, 3]

        c2 = torch.cos(q2)
        m11 = self.a1 + self.a2 + 2 * self.a3 * c2
        m12 = self.a2 + self.a3 * c2
        m21 = m12
        m22 = (
            torch.ones((batch_size)) * self.a2
        )  # Need to broadcast a2 term onto the batch

        m = self.stack_2x2matrices((m11, m12, m21, m22))
        im = torch.inverse(m)

        c = self.stack_2x2matrices(
            (-2 * q2_dot, -q2_dot, q1_dot, torch.zeros(batch_size))
        )
        c = self.a3 * torch.sin(q2) * c  # broadcasting

        g11 = self.a4 * torch.sin(q1) + self.a5 * torch.sin(q1 + q2)
        g21 = self.a5 * torch.sin(q1 + q2)
        g = torch.stack([g11, g21], dim=1)

        q_ddot = torch.bmm(
            im, (-torch.bmm(c, q_dot.unsqueeze(2)).squeeze(2) - g + u).unsqueeze(2)
        ).squeeze(2)
        dx = torch.cat((q_dot, q_ddot), dim=1)
        y = q
        return dx, y

    def stack_2x2matrices(self, m: tuple):
        m11, m12, m21, m22 = m
        return torch.stack(
            [torch.stack([m11, m12], dim=1), torch.stack([m21, m22], dim=1)], dim=1
        )

    def clone(self):
        copy = type(self)()
        copy.load_state_dict(self.state_dict())
        return copy
