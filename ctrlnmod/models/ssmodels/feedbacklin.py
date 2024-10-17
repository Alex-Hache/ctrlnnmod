import torch.nn as nn
import torch
from collections import OrderedDict
from .linear import NnLinear
from ...layers.layers import BetaLayer
from typing import Optional, Literal
from ctrlnmod.models.ssmodels import H2BoundedLinear, L2BoundedLinear
from ctrlnmod.models.feedforward import LBDN, FFNN
from torch import Tensor


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
        dist_dim: int = 0,
        linear_model: Optional[Literal["L2", "H2"]] = None,
        gamma: Optional[float] = None,
        actF: str = 'tanh',
        bias: bool = True,
        alpha: float = 0.0,
        lambda_alpha: Optional[float] = None
    ):
        r"""
        Constructor for FLNSSM Elman models with optional bounded disturbances
        z^+ = Az + B(u + \alpha(y,d)) + Gd
        y = Cz

        params:
        * input_dim : size of control input
        * hidden_dim : size of hidden layers
        * state_dim : size of the state-space
        * output_dim : size of the output layer
        * n_hid_layers : number of hidden layers
        * dist_dim : size of disturbance input (optional)
        * linear_model : type of linear model for disturbances ("L2", "H2", or None for unconstrained)
        * gamma : prescribed L2 gain or H2 norm for disturbances (if applicable)
        * actF : activation function for nonlinear residuals
        * bias : whether to use bias in linear layers
        * alpha : alpha stability (only for L2 model)
        """
        super(FLNSSM_Elman, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers = n_hid_layers
        self.dist_dim = dist_dim
        self.bias = bias

        if actF.lower() == 'tanh':
            self.act_f = nn.Tanh()
        elif actF.lower() == 'relu':
            self.act_f = nn.ReLU()
        else:
            raise NotImplementedError(
                f"Function {actF} not implemented. Choose 'tanh' or 'relu'.")
        self.act_name = actF

        # If there are disturbances
        if dist_dim > 0:
            # Linear part
            if linear_model is None:
                self.linmod = NnLinear(
                    input_dim + dist_dim, state_dim, output_dim)
            elif linear_model == "L2":
                if gamma is not None:
                    self.linmod = L2BoundedLinear(
                        dist_dim, output_dim, state_dim, gamma, alpha)
                else:
                    raise ValueError("Please specify a value for gamma")
            elif linear_model == "H2":
                if gamma is not None:
                    self.linmod = H2BoundedLinear(
                        dist_dim, output_dim, state_dim, gamma)
                else:
                    raise ValueError("Please specify a value for gamma")
            else:
                raise ValueError(
                    "linear_model must be either 'L2', 'H2', or None")
            # Control input matrix
            self.B = nn.Parameter(torch.randn(state_dim, input_dim))
        elif dist_dim == 0:
            # Linear part
            if linear_model is None:
                self.linmod = NnLinear(input_dim, state_dim, output_dim)
            elif linear_model == "L2":
                if gamma is not None:
                    self.linmod = L2BoundedLinear(
                        input_dim, output_dim, state_dim, gamma, alpha)
                else:
                    raise ValueError("Please specify a value for gamma")
            elif linear_model == "H2":
                if gamma is not None:
                    self.linmod = H2BoundedLinear(
                        input_dim, output_dim, state_dim, gamma)
                else:
                    raise ValueError("Please specify a value for gamma")
            else:
                raise ValueError(
                    "linear_model must be either 'L2', 'H2', or None")
        else:
            raise ValueError(
                f"Distubrances dimensions must be nonnegative found {dist_dim}")

        if lambda_alpha is not None:  # An upper bound on the lipschitz constant is prescribed
            self.alpha = LBDN(output_dim + dist_dim, hidden_dim, input_dim, torch.tensor(
                [lambda_alpha] * (output_dim + dist_dim)), act_f=self.act_f)
        else:

            self.alpha = FFNN(output_dim + dist_dim,
                              hidden_dim, input_dim, act_f=self.act_f)

        self.init_weights_()

    def forward(self, input: Tensor, state: Tensor):
        u = input[:, :self.input_dim]
        z = state

        if self.dist_dim > 0:
            d = input[:, :self.input_dim]
            A, G, C = self.linmod._frame()
        else:
            A, B, C = self.linmod._frame()

        y = C @ z

        if self.dist_dim > 0:
            alpha = self.alpha(torch.cat((y, d), dim=1))
            dz = z @ A.T + (u + alpha) @ self.B.T + d @ G.T
        else:
            alpha = self.alpha(y)
            dz = z @ A.T + (u + alpha) @ B.T

        return dz, y

    def init_weights_(self, A0=None, B0=None, C0=None, G0=None, isLinTrainable=True):
        # Initialisation des poids linéaires
        if A0 is not None and B0 is not None and C0 is not None:
            if isinstance(self.linmod, NnLinear):
                self.linmod.init_model_(A0, torch.cat(
                    (B0, G0), dim=1) if G0 is not None else B0, C0, requires_grad=isLinTrainable)
            elif isinstance(self.linmod, L2BoundedLinear):
                self.linmod.right_inverse_(A0, G0, C0, float(
                    self.linmod.gamma), float(self.linmod.alpha))
            elif isinstance(self.linmod, H2BoundedLinear):
                self.linmod.right_inverse_(
                    A0, G0, C0, float(self.linmod.gamma2))
            self.B.data = B0
            if isinstance(self.linmod, NnLinear) and self.dist_dim > 0 and G0 is not None:
                self.G.data = G0
        else:
            # Initialisation par défaut si les matrices ne sont pas fournies
            nn.init.xavier_uniform_(self.B)
            if isinstance(self.linmod, NnLinear) and self.dist_dim > 0:
                nn.init.xavier_uniform_(self.G)

        # Initialisation des poids non linéaires
        if not self.use_lipschitz:
            # Pour les réseaux standard
            nn.init.xavier_uniform_(
                self.alpha_in_d.weight, gain=nn.init.calculate_gain(self.act_name))
            nn.init.xavier_uniform_(
                self.alpha_in_y.weight, gain=nn.init.calculate_gain(self.act_name))

            if self.n_hid_layers > 1:
                # Exclure la couche d'entrée et de sortie
                for m in self.alpha_hid[1:-1]:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(
                            m.weight, gain=nn.init.calculate_gain(self.act_name))

            nn.init.zeros_(self.alpha_out.weight)
            if self.alpha_out.bias is not None:
                nn.init.zeros_(self.alpha_out.bias)
        else:
            # Pour les réseaux bornés par Lipschitz
            # Ces réseaux ont généralement leur propre méthode d'initialisation
            # qui respecte les contraintes de Lipschitz, donc nous ne les modifions pas ici
            pass

    def load_weights(self, params_dict):
        self.linmod.load_state_dict(params_dict["linmod"])
        self.B.data = torch.Tensor(params_dict["B"])
        if isinstance(self.linmod, NnLinear) and self.dist_dim > 0:
            self.G.data = torch.Tensor(params_dict["G"])
        self.alpha_in_y.weight = nn.parameter.Parameter(
            torch.Tensor(params_dict["alpha_in_y"]))
        self.alpha_in_y.bias = nn.parameter.Parameter(
            torch.Tensor(params_dict["b_in"]).squeeze())
        if self.dist_dim > 0:
            self.alpha_in_d.weight = nn.parameter.Parameter(
                torch.Tensor(params_dict["alpha_in_d"]))
        self.alpha_out.weight = nn.parameter.Parameter(
            torch.Tensor(params_dict["alpha_out"]))
        if self.alpha_out.bias is not None:
            self.alpha_out.bias = nn.parameter.Parameter(
                torch.Tensor(params_dict["b_out"]).squeeze())

    def clone(self):
        linear_model = None
        gamma = None
        alpha = 0.0
        if isinstance(self.linmod, L2BoundedLinear):
            linear_model = "L2"
            gamma = float(self.linmod.gamma)
            alpha = float(self.linmod.alpha)
        elif isinstance(self.linmod, H2BoundedLinear):
            linear_model = "H2"
            gamma = float(self.linmod.gamma2)

        copy = type(self)(
            self.input_dim,
            self.hidden_dim,
            self.state_dim,
            self.output_dim,
            self.n_hid_layers,
            self.dist_dim,
            linear_model,
            gamma,
            self.act_name,
            self.bias,
            alpha
        )
        copy.load_state_dict(self.state_dict())
        return copy

    def __repr__(self):
        linear_type = "Unconstrained"
        gamma = "N/A"
        if isinstance(self.linmod, L2BoundedLinear):
            linear_type = "L2"
            gamma = self.linmod.gamma
        elif isinstance(self.linmod, H2BoundedLinear):
            linear_type = "H2"
            gamma = self.linmod.gamma2
        return (f"FLNSSM_Elman: input_dim={self.input_dim}, state_dim={self.state_dim}, "
                f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
                f"n_hid_layers={self.n_hid_layers}, dist_dim={self.dist_dim}, "
                f"activation={self.act_name}, linear_model={linear_type}, gamma={gamma}")

    def __str__(self):
        return "FLNSSM_Elman"

    def check(self):
        if not isinstance(self.linmod, NnLinear):
            return self.linmod.check_()
        return True, None


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
        self.alpha_out = nn.Linear(self.hidden_dim, self.input_dim, bias=False)

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
