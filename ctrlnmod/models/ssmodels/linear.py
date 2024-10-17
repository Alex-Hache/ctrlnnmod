from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from geotorch_custom.parametrize import is_parametrized
import torch
from torch import Tensor


class NnLinear(Module):
    """
    neural network module corresponding to a linear state-space model
        x^+ = Ax + Bu
        y = Cx
    """

    def __init__(
        self, input_dim: int, output_dim: int, state_dim: int, alpha=None
    ) -> None:
        super(NnLinear, self).__init__()

        self.nu = input_dim
        self.nx = state_dim
        self.ny = output_dim
        self.str_savepath = "./results"
        self.A = Linear(self.nx, self.nx, bias=False)
        self.B = Linear(self.nu, self.nx, bias=False)
        self.C = Linear(self.nx, self.ny, bias=False)
        self.alpha = alpha

        # Is A alpha stable ?
        if alpha is not None:
            geo.alpha_stable(self.A, 'weight', alpha=alpha)

    def __repr__(self):
        if is_parametrized(self.A):
            return "Stable_Linear_ss" + f"_{self.alpha}"
        else:
            return "Linear_ss"

    def forward(self, u, x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx, y

    def _frame(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.A.weight, self.B.weight, self.C.weight

    def eval_(self):
        return self.A.weight, self.B.weight, self.C.weight

    def right_inverse_(self, A0, B0, C0, requires_grad=True):
        # Check matrix dimensions
        assert self.nx == A0.shape[0] == A0.shape[
            1], f"Given A matrix has incorrect size: nx = {self.nx}, found {A0.shape}"
        assert (
            self.nx, self.nu) == B0.shape, f"Given B matrix has incorrect size: expected ({self.nx}, {self.nu}), found {B0.shape}"
        assert (
            self.ny, self.nx) == C0.shape, f"Given C matrix has incorrect size: expected ({self.ny}, {self.nx}), found {C0.shape}"

        # Initialize A
        if is_parametrized(self.A):
            self.A.weight = A0
        else:
            self.A.weight = Parameter(A0)

        # Initialize B and C
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)

        self.A0 = self.A.weight.detach().clone()
        self.B0 = self.B.weight.detach().clone()
        self.C0 = self.C.weight.detach().clone()

        # Set requires_grad
        if not requires_grad:
            if is_parametrized(self.A):
                for parameter in self.A.parameters():
                    parameter.requires_grad_(False)
            else:
                self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def check_(self):
        if self.alpha is None:
            alpha = 0.0
        else:
            alpha = self.alpha
        eig_vals = torch.real(torch.linalg.eigvals(self.A.weight))
        return torch.all(eig_vals <= alpha), torch.max(eig_vals)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def init_weights_(self, A0, B0, C0, requires_grad=True, margin=0.1, adjust_alpha=False):
        # Compute the Lyapunov exponent of A0
        A0_lyap_exp = -torch.max(torch.real(torch.linalg.eigvals(A0)))

        if adjust_alpha:  # For now to be set to fals
            # Adjust alpha to be slightly smaller than A0's Lyapunov exponent
            self.alpha = torch.tensor(
                float(A0_lyap_exp) - margin, device=A0.device)
        else:
            # Shift eigenvalues of A0 to match the desired alpha
            if self.alpha is None:
                alpha = 0.0
            else:
                alpha = self.alpha
            if A0_lyap_exp < alpha:
                shift = self.alpha - A0_lyap_exp + margin
                # Move to left part of the left half plane
                A0 = A0 - shift * torch.eye(self.nx, device=A0.device)

        self.right_inverse_(A0, B0, C0, requires_grad=requires_grad)


'''

class HinfNN(Module):
    def __init__(self, input_dim: int, output_dim: int, config) -> None:
        super(HinfNN, self).__init__()
        self.nu = input_dim
        self.nx = config.nx
        self.ny = output_dim
        self.gamma = config.gamma
        self.config = config

        self.Q = Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.P = Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.G = Parameter(nn.init.normal_(torch.empty((self.ny, self.nx))))
        self.H = Parameter(
            nn.init.normal_(torch.empty((self.nx, self.nu)))
        )  # H is restriction to input space
        self.eps = torch.Tensor([1e-4])
        # register P,Q,S variables to be on specified manifolds
        geo.positive_definite(self, "P")
        geo.positive_definite(self, "Q")
        geo.skew_symmetric(self, "S")

    def forward(self, u, x):

        # Building linear matrix system
        Htilde = self.H / (1.01 * torch.sqrt(torch.norm(self.H @ self.H.T, 2)))
        Asym = -0.5 * (self.Q + self.G.T @ self.G + self.eps * torch.eye(self.nx))
        A = (Asym + self.S) @ self.P
        B = self.gamma * sqrtm(self.Q) @ Htilde
        C = self.G @ self.P

        dx = x @ A.T + u @ B.T
        y = x @ C.T
        return dx, y


class Grnssm(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        actF=Tanh(),
        out_eq_nl=False,
        alpha=None,
    ) -> None:
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
        copy = type(self)(
            self.nu,
            self.ny,
            self.nx,
            self.nh,
            self.n_hid_layers,
            self.actF,
            self.out_eq_nl,
        )
        copy.load_state_dict(self.state_dict())
        return copy


class LipGrnssm(Grnssm):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        actF=Tanh(),
        out_eq_nl=False,
        lip: Tuple[int, int] = (1, 1),
    ) -> None:
        super(LipGrnssm, self).__init__(
            input_dim,
            output_dim,
            state_dim,
            hidden_dim,
            n_hidden_layers,
            actF,
            out_eq_nl,
        )

        # We override the definition of the nonlinear parts of the network
        # and replace them with LBDN networks

        # Nonlinear state part
        self.lip_x = lip[0]
        self.lip_u = lip[1]

        self.fx = LipFxu(
            self.nu, self.nh, self.nx, scalex=self.lip_x, scaleu=self.lip_u
        )
        # Nonlinear output part
        if self.out_eq_nl:
            self.hx = LipHx(self.nx, self.nh, self.ny, scalex=self.lip_x)

    def forward(self, u, x):
        return super().forward(u, x)

    def init_weights_(self, A0, B0, C0, isLinTrainable=True):
        super().init_weights_(A0, B0, C0, isLinTrainable)


class L2IncGrnssm(Grnssm):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        actF=Tanh(),
        out_eq_nl=False,
        alpha=None,
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            state_dim,
            hidden_dim,
            n_hidden_layers,
            actF,
            out_eq_nl,
            alpha,
        )

'''
