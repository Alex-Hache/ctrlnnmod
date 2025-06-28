import torch.nn as nn
import torch
from collections import OrderedDict
from .linear import SSLinear
from ...layers.layers import BetaLayer
from typing import Optional, Literal, Tuple
from ctrlnmod.models.ssmodels import H2BoundedLinear, L2BoundedLinear
from ctrlnmod.models.feedforward import LBDN, FFNN
from torch import Tensor
from ctrlnmod.utils import FrameCacheManager
from abc import ABC, abstractmethod


# TODO : make a base class for linearizable nn with or without decouplig matrix decoupling matrix construction must be optional.
class FLNSSM_decoupling(nn.Module):
    def __init__(
        self,
        nu: int,
        nh: int,
        nx: int,
        ny: int,
        n_hid_layers: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, x]:
            z^+ = Az + B*beta(x)(u + alpha(x))
            y = Cz

        params :
            * nu : size of control input
            * nh : size of hidden layers
            * nx : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * ny : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(FLNSSM_decoupling, self).__init__()

        # Set network dimensions
        self.nu = nu
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.n_hid_layers = n_hid_layers

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = SSLinear(self.nu, self.nx, self.ny)
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Beta layer initialized to I_nu
        self.beta = BetaLayer(
            self.nu, self.nx, self.nh, self.actF
        )

        # Alpha layer

        self.alpha_in = nn.Linear(self.nx, self.nh)
        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(nh, nh))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.nh, self.nx, bias=False)

        # Nonlinear part initialized to 0
        nn.init.zeros_(self.alpha_out.weight)

    def forward(self, input, state):
        """
        params :
            - input = [u, x]
            - state =  state vector
        """
        u = input[:, 0: self.nu]
        x = input[:, self.nu: self.nu + self.nx]
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
            self.nu,
            self.nh,
            self.nx,
            self.ny,
            self.n_hid_layers,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy


class QFLNSSM(nn.Module):
    def __init__(
        self,
        nu: int,
        nh: int,
        nx: int,
        ny: int,
        n_hid_layers: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, x]:
            z^+ = Az + B*beta(x)(u + alpha(x)) + g(z,u)
            y = Cz

        params :
            * nu : size of control input
            * nh : size of hidden layers
            * nx : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * ny : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(QFLNSSM, self).__init__()

        # Set network dimensions
        self.nu = nu
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.n_hid_layers = n_hid_layers

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = SSLinear(self.nu, self.nx, self.ny)
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Beta layer by default it is initialized to the identity
        self.beta = BetaLayer(
            self.nu, self.nx, self.nh, self.actF
        )

        # Alpha layer

        self.alpha_in = nn.Linear(self.nx, self.nh)
        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(nh, nh))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.nh, self.nu, bias=False)

        # Nonlinear part initialized to 0
        nn.init.zeros_(self.alpha_out.weight)

        # g(z,u)
        self.g_in_z = nn.Linear(self.nx, self.nh)
        self.g_in_u = nn.Linear(self.nu, self.nh)
        if self.n_hid_layers > 1:
            paramsNLHidG = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(nh, nh))
                paramsNLHidG.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidG.append(tupact)
            self.g_hid = nn.Sequential(OrderedDict(paramsNLHidG))
        self.g_out = nn.Linear(self.nh, self.nx, bias=False)

    def forward(self, input, state):
        """
        params :
            - input = [u, x]
            - state =  state vector
        """
        u = input[:, 0: self.nu]
        x = input[:, self.nu: self.nu + self.nx]
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
            self.nu,
            self.nh,
            self.nx,
            self.ny,
            self.n_hid_layers,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy



class FLNSSM_Base(nn.Module, ABC):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hid_layers: int,
        linear_model: Optional[Literal["L2", "H2"]] = None,
        gamma: Optional[float] = None,
        actF: str = 'tanh',
        bias: bool = True,
        alpha: float = 0.0,
        lambda_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.nu = nu
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.n_hid_layers = n_hid_layers
        self.bias = bias
        
        # Activation function setup
        if actF.lower() == 'tanh':
            self.act_f = nn.Tanh()
        elif actF.lower() == 'relu':
            self.act_f = nn.ReLU()
        else:
            raise NotImplementedError(f"Function {actF} not implemented. Choose 'tanh' or 'relu'.")
        self.act_name = actF
        
        # Setup linear model based on type
        self._setup_linear_model(linear_model, gamma, alpha)
        
        # Setup alpha network
        self._setup_alpha_network(lambda_alpha)
        
        self._frame_cache = FrameCacheManager()
    
    @abstractmethod
    def _setup_linear_model(self, linear_model, gamma, alpha):
        pass
    
    @abstractmethod
    def _setup_alpha_network(self, lambda_alpha):
        pass
    
    @abstractmethod
    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        pass
    
    @abstractmethod
    def _frame(self) -> Tuple[Tensor, ...]:
        pass


class FLNSSM_Jordan_Standard(FLNSSM_Base):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hid_layers: int,
        linear_model: Optional[Literal["L2", "H2"]] = None,
        gamma: Optional[float] = None,
        actF: str = 'tanh',
        bias: bool = True,
        alpha: float = 0.0,
        lambda_alpha: Optional[float] = None,
    ):
        super().__init__(nu, ny, nx, nh, n_hid_layers,linear_model,
                         gamma, actF, bias, alpha, lambda_alpha)
    
    def _setup_linear_model(self, linear_model, gamma, alpha):
        if linear_model is None:
            self.linmod = SSLinear(self.nu, self.nx, self.ny)
        elif linear_model == "L2":
            if gamma is None:
                raise ValueError("Please specify a value for gamma")
            self.linmod = L2BoundedLinear(self.nu, self.ny, self.nx, gamma, alpha)
        elif linear_model == "H2":
            if gamma is None:
                raise ValueError("Please specify a value for gamma")
            self.linmod = H2BoundedLinear(self.nu, self.ny, self.nx, gamma)
        else:
            raise ValueError("linear_model must be either 'L2', 'H2', or None")
    
    def _setup_alpha_network(self, lambda_alpha):
        if lambda_alpha is not None:
            self.alpha = LBDN(self.ny, self.nh, self.nu, 
                            torch.tensor([lambda_alpha] * self.ny),
                            act_f=self.act_f, bias=self.bias, n_hidden=self.n_hid_layers)
        else:
            self.alpha = FFNN(self.ny, self.nh, self.nu,
                            act_f=self.act_f, bias=self.bias, n_hidden=self.n_hid_layers)
    
    def forward(self, input: Tensor, state: Tensor):
        z = state
        u = input
        A, B, C, _ = self._frame()
        
        y = z @ C.T
        alpha = self.alpha(y)
        dz = z @ A.T + (u + alpha) @ B.T
        
        return dz, y
    
    def _frame(self):
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        A, B, C = self.linmod._frame()
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, B, C, torch.zeros((self.ny, self.nu)))
        
        return A, B, C, torch.zeros((self.ny, self.nu))

class FLNSSM_Jordan_Disturbed(FLNSSM_Base):
    def __init__(self, *args, dist_dim: int, **kwargs):
        self.nd = dist_dim
        if dist_dim <= 0:
            raise ValueError(f"Disturbance dimension must be positive, found {dist_dim}")
        super().__init__(*args, **kwargs)
        self.B = nn.Parameter(torch.randn(self.nx, self.nu))
    
    def _setup_linear_model(self, linear_model, gamma, alpha):
        if linear_model is None:
            self.linmod = SSLinear(self.nd, self.ny, self.nx)
        elif linear_model == "L2":
            if gamma is None:
                raise ValueError("Please specify a value for gamma")
            self.linmod = L2BoundedLinear(self.nu, self.ny, self.nx, gamma=gamma, alpha=alpha, nd =self.nd,)
        elif linear_model == "H2":
            if gamma is None:
                raise ValueError("Please specify a value for gamma")
            self.linmod = H2BoundedLinear(self.nd, self.ny, self.nx, gamma)
        else:
            raise ValueError("linear_model must be either 'L2', 'H2', or None")
    
    def _setup_alpha_network(self, lambda_alpha):
        input_dim = self.ny + self.nd
        if lambda_alpha is not None:
            self.alpha = LBDN(input_dim, self.nh, self.nu,
                            torch.tensor([lambda_alpha] * input_dim),
                            act_f=self.act_f, bias=self.bias, n_hidden=self.n_hid_layers)
        else:
            self.alpha = FFNN(input_dim, self.nh, self.nu,
                            act_f=self.act_f, bias=self.bias, n_hidden=self.n_hid_layers)
    
    def forward(self, input: Tensor, state: Tensor):
        z = state
        u = input[:, :self.nu]
        d = input[:, self.nu:]
        
        assert d.shape[1] == self.nd
        
        A, G, C = self._frame()
        
        y = z @ C.T
        alpha = self.alpha(torch.cat((y, d), dim=1))
        dz = z @ A.T + (u + alpha) @ self.B.T + d @ G.T
        
        return dz, y
    
    def _frame(self):
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        A, G, C = self.linmod._frame()
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, G, C)
        
        return A, G, C

    def init_weights_(self, A0, B0, C0, gamma, alpha, G0=None):
        # Initializing linear weights
        self.linmod.init_weights_(A0, B0, C0, gamma=gamma, alpha=alpha, G0=G0)

        # Initializing nonlinear output weights to 0
        nn.init.zeros_(self.alpha.Wout.weight)
        if self.alpha.Wout.bias is not None:
            nn.init.zeros_(self.alpha.Wout.bias)

class FLNSSM_Jordan(nn.Module):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hid_layers: int,
        dist_dim: int = 0,
        linear_model: Optional[Literal["L2", "H2"]] = None,
        gamma: Optional[float] = None,
        actF: str = 'tanh',
        bias: bool = True,
        alpha: float = 0.0,
        lambda_alpha: Optional[float] = None,
        **kwargs
    ):
        r"""
        Constructor for FLNSSM Jordan models with optional bounded disturbances
        z^+ = Az + B(u + \alpha(y,d)) + Gd
        y = Cz

        params:
        * nu : size of control input
        * nh : size of hidden layers
        * nx : size of the state-space
        * ny : size of the output layer
        * n_hid_layers : number of hidden layers
        * dist_dim : size of disturbance input (optional)
        * linear_model : type of linear model for disturbances ("L2", "H2", or None for unconstrained)
        * gamma : prescribed L2 gain or H2 norm for disturbances (if applicable)
        * actF : activation function for nonlinear residuals
        * bias : whether to use bias in linear layers
        * alpha : alpha stability (only for L2 model)
        """
        super(FLNSSM_Jordan, self).__init__()

        self.nu = nu
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.n_hid_layers = n_hid_layers
        self.nd = dist_dim
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
                self.linmod = SSLinear(
                    dist_dim, ny, nx)
            elif linear_model == "L2":
                if gamma is not None:
                    self.linmod = L2BoundedLinear(
                        dist_dim, ny, nx, gamma, alpha)
                else:
                    raise ValueError("Please specify a value for gamma")
            elif linear_model == "H2":
                if gamma is not None:
                    self.linmod = H2BoundedLinear(
                        dist_dim, ny, nx, gamma)
                else:
                    raise ValueError("Please specify a value for gamma")
            else:
                raise ValueError(
                    "linear_model must be either 'L2', 'H2', or None")
            # Control input matrix
            self.B = nn.Parameter(torch.randn(nx, nu))
        elif dist_dim == 0:
            # Linear part
            if linear_model is None:
                self.linmod = SSLinear(nu, nx, ny)
            elif linear_model == "L2":
                if gamma is not None:
                    self.linmod = L2BoundedLinear(
                        nu, ny, nx, gamma, alpha)
                else:
                    raise ValueError("Please specify a value for gamma")
            elif linear_model == "H2":
                if gamma is not None:
                    self.linmod = H2BoundedLinear(
                        nu, ny, nx, gamma)
                else:
                    raise ValueError("Please specify a value for gamma")
            else:
                raise ValueError(
                    "linear_model must be either 'L2', 'H2', or None")
        else:
            raise ValueError(
                f"Distubrances dimensions must be nonnegative found {dist_dim}")

        if lambda_alpha is not None:  # An upper bound on the lipschitz constant is prescribed
            self.alpha = LBDN(ny + dist_dim, nh, nu, torch.tensor(
                [lambda_alpha] * (ny + dist_dim)), act_f=self.act_f, bias=bias)
        else:

            self.alpha = FFNN(ny + dist_dim,
                              nh, nu, act_f=self.act_f, bias=bias)

        self.init_weights_()

        self._frame_cache = FrameCacheManager()

    def _frame(self):
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache

        A, B, C, D = *self.linmod._frame(), torch.zeros((self.ny, self.nd))
        # Stocker dans le cache si la mise en cache est active
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, B, C, D)

        return A, B, C, D 


    def forward(self, input: Tensor, state: Tensor):
        z = state            
        u = input[:, :self.nu]
        d = input[:, self.nu:]
        if self.nd > 0:
            A, G, C, D = self._frame()
        else:
            A, B, C, D = self._frame()

        y = z @ C.T + d @ D.T

        if self.nd > 0 :
            alpha = self.alpha(torch.cat((y, d), dim=1))
            dz = z @ A.T + (u + alpha) @ self.B.T + d @ G.T
        else:
            alpha = self.alpha(y)
            dz = z @ A.T + (u + alpha) @ B.T

        return dz, y

    def init_weights_(self, A0=None, B0=None, C0=None, G0=None, isLinTrainable=True):
        # Initialisation des poids linéaires
        if A0 is not None and B0 is not None and C0 is not None:
            if isinstance(self.linmod, SSLinear):
                self.linmod.init_weights_(A0, G0, C0, requires_grad=isLinTrainable)
            elif isinstance(self.linmod, L2BoundedLinear):
                self.linmod.right_inverse_(A0, G0, C0, float(
                    self.linmod.gamma), float(self.linmod.alpha))
            elif isinstance(self.linmod, H2BoundedLinear):
                self.linmod.right_inverse_(
                    A0, G0, C0, float(self.linmod.gamma2))
            self.B.data = B0
        else:
            # Initialisation par défaut si les matrices ne sont pas fournies
            nn.init.xavier_uniform_(self.B)
        self.alpha.init_weights_()

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
            self.nu,
            self.nh,
            self.nx,
            self.ny,
            self.n_hid_layers,
            self.nd,
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
        return (f"FLNSSM_Jordan: nu={self.nu}, nx={self.nx}, "
                f"nh={self.nh}, ny={self.ny}, "
                f"n_hid_layers={self.n_hid_layers}, dist_dim={self.nd}, "
                f"activation={self.act_name}, linear_model={linear_type}, gamma={gamma}")

    def __str__(self):
        return "FLNSSM_Jordan"

    def check(self):
        if not isinstance(self.linmod, NnLinear):
            return self.linmod.check()
        return True, None


class FLNSSM_Jordan_Dist(nn.Module):
    def __init__(
        self,
        nu: int,
        nh: int,
        nx: int,
        ny: int,
        n_hid_layers: int,
        nd: int,
        actF=nn.Tanh(),
    ):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, y, d]:
            z^+ = Az + B(u + alpha(y, d)) + Gd
            y = Cz

        params :
            * nu : size of control input
            * nh : size of hidden layers
            * nx : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * ny : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(FLNSSM_Jordan_Dist, self).__init__()

        # Set network dimensions
        self.nu = nu
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.n_hid_layers = n_hid_layers
        self.dist_dim = nd

        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = SSLinear(
            self.nu + self.dist_dim, self.nx, self.ny
        )
        # Nonlinear part for the state beta(x)(u+alpha(x))

        # Alpha layer
        self.alpha_in_y = nn.Linear(
            self.ny, self.nh, bias=True)
        self.alpha_in_d = nn.Linear(self.dist_dim, self.nh, bias=False)

        if self.n_hid_layers > 1:
            paramsNLHidA = []
            for k in range(n_hid_layers - 1):
                tup = ("dense{}".format(k), nn.Linear(nh, nh))
                paramsNLHidA.append(tup)
                tupact = ("actF{}".format(k), actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.nh, self.nu, bias=False)

    def forward(self, input, state):
        """
        params :
            - input = [u, d]
            - state =  state vector
        """
        u = input[:, 0: self.nu]
        d = input[:, self.nu: self.nu + self.dist_dim]

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
        self.linmod.init_weights_(A0, B0, C0, requires_grad=isLinTrainable)

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
            self.nu,
            self.nh,
            self.nx,
            self.ny,
            self.n_hid_layers,
            self.dist_dim,
            self.actF,
        )
        copy.load_state_dict(self.state_dict())
        return copy
