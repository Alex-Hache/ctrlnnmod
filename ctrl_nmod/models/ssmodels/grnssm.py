from torch.nn import Module
from ctrl_nmod.models.feedforward.lbdn import Fxu, LipFxu, LipHx, Hx
from torch.nn.init import zeros_
from torch.linalg import eigvals
from torch import Tensor, real, min
from typing import Tuple
from ctrl_nmod.models.ssmodels.linear import NnLinear
from ctrl_nmod.models.ssmodels.hinf import L2BoundedLinear
import torch
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.atoms.affine.bmat import bmat
import numpy as np
from math import sqrt
from scipy.io import savemat
import torch.nn as nn


class Grnssm(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        actF: str = 'tanh',
        out_eq_nl=False,
        alpha=None,
    ) -> None:
        r"""
        u is a generalized input ex : [control, distrubance]:
            .. ::math
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
        self.act_name = actF  # Name of the activation function

        # Activation functions
        if actF.lower() == 'tanh':
            self.actF = nn.Tanh()
        elif actF.lower() == 'relu':
            self.actF = nn.ReLU()
        else:
            raise NotImplementedError(
                f"Function {actF} not yet implemented please choose from 'tanh' or 'relu' ")

        # Nonlinear output equation
        self.out_eq_nl = out_eq_nl

        # Linear part
        self.linmod = NnLinear(self.nu, self.ny, self.nx, alpha=alpha)
        self.alpha = alpha

        # Nonlinear state part
        self.fx = Fxu(self.nu, self.nh, self.nx)
        if self.out_eq_nl:
            self.hx = Hx(self.nx, self.nh, self.ny)

    def __repr__(self):
        return f"GRNSSM : nu={self.nu} nx={self.nx} nh={self.nh} ny={self.ny} activation = {self.act_name}"

    def __str__(self) -> str:
        return "GRNSSM"

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
        """
            It can be used to initialize the linear part of the model.
            For example using linear subspace methods or Best Linear Approximation
        """

        self.linmod.init_model_(A0, B0, C0, requires_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        zeros_(self.fx.Wout.weight)
        if self.fx.Wout.bias is not None:
            zeros_(self.fx.Wout.bias)
        # zeros_(self.Wh.weight)
        if self.out_eq_nl:
            zeros_(self.hx.Wout.weight)
            if self.hx.Wout.bias is not None:
                zeros_(self.hx.Wout.bias)

        nn.init.xavier_uniform_(
            self.fx.Wfu, gain=nn.init.calculate_gain(self.act_name))
        nn.init.xavier_uniform_(
            self.fx.Wfx, gain=nn.init.calculate_gain(self.act_name))

    def clone(self):  # Method called by the simulator
        copy = type(self)(
            self.nu,
            self.ny,
            self.nx,
            self.nh,
            self.n_hid_layers,
            self.act_name,
            self.out_eq_nl,
        )
        copy.load_state_dict(self.state_dict())
        return copy

    def check_(self, *args):
        return True


class LipGrnssm(Grnssm):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        actF: str = 'tanh',
        out_eq_nl=False,
        lip: Tuple[Tensor, Tensor] = (Tensor([1]), Tensor([1])),
        alpha=None
    ) -> None:
        super(LipGrnssm, self).__init__(
            input_dim,
            output_dim,
            state_dim,
            hidden_dim,
            n_hidden_layers,
            actF,
            out_eq_nl,
            alpha=alpha
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

    def right_inverse_(self, A0, B0, C0, isLinTrainable=True):
        super().init_weights_(A0, B0, C0, isLinTrainable)

    def check_(self) -> bool:
        return (self.fx.check_() and self.hx.check_()) if self.out_eq_nl else self.fx.check_()


class StableGNSSM(LipGrnssm):
    r"""
        x_dot = Ax + Bu + f(x,u)
        y = Cx + (h(x))
        avec $A-(\lambda_x + \epsilon)$ stable et $\lambda$ Lipschitz selon x
    """

    def __init__(self, nu: int, ny: int, nx: int,
                 nh: int, n_hidden_layers: int = 1, actF='tanh',
                 out_eq_nl=False, lmbd=1.0, epsilon=1e-2):
        super(StableGNSSM, self).__init__(nu, ny, nx, nh, n_hidden_layers, actF, out_eq_nl,
                                          lip=(Tensor([lmbd]), Tensor([10])), alpha=lmbd + epsilon)


class L2IncGrNSSM(LipGrnssm):
    def __init__(self, nu: int, ny: int, nx: int,
                 nh: int, n_hidden_layers: int = 1, actF='tanh',
                 out_eq_nl=False, l2i=1.0, alpha=1.0) -> None:
        # Compute necessary bound for L2i according to Lipschitz upper bound for nl part
        # epsilon_u = torch.sqrt([1.0])
        # epsilon_x = torch.sqrt([1.0])
        self.gamma = l2i
        lipx = torch.sqrt(torch.Tensor([2 * alpha]))  # Dummy init
        lip = (lipx, l2i / torch.sqrt(torch.Tensor([2])))
        super().__init__(nu, ny, nx, nh, n_hidden_layers, actF, out_eq_nl, lip, alpha)
        scaleH = 1 / sqrt(2) - 0.1
        self.linmod = L2BoundedLinear(
            nu, ny, nx, gamma=l2i, alpha=alpha, scaleH=scaleH, epsilon=2.0)
        self.frame_()

    def __repr__(self):
        return f"Incr L2 bounded GRNSSM : eta={float(self.gamma)} -- nh={self.nh}"

    def frame_(self):
        self.alpha = (self.lip_x**2) / (2 * min(real(eigvals(self.linmod.P))))
        self.linmod.alpha = self.alpha

    def forward(self, u, x):
        return super().forward(u, x)

    def right_inverse_(self, A, B, C, eta, alpha):
        self.linmod.right_inverse_(A, B, C, eta, alpha)

    def check_(self, traj=None, N_trys=1000) -> Tuple[bool, np.ndarray]:
        self.frame_()
        gammas = []
        if traj is not None:
            u_traj, x_traj = traj

            for u_eq, x_eq in zip(u_traj, x_traj):
                gamma = self.compute_L2_taylor(
                    u_eq.unsqueeze(0), x_eq.unsqueeze(0))
                gammas.append(gamma)
        else:
            for k in range(N_trys):  # Compute around the origin
                u_eq, x_eq = torch.rand((1, self.nu)), torch.rand((1, self.nx))
                gamma = self.compute_L2_taylor(u_eq, x_eq)
                gammas.append(gamma)
        bl2i = all(np.array(gammas) <= gamma)
        b_lin_bounded, _ = self.linmod.check_()
        if not b_lin_bounded:
            print("Prescribed L2 gain for linear part not okay")
        bLipschitz = self.fx.check_()
        if not bLipschitz:
            print("Prescribed Lipschitz constant for nonlinear part not okay")

        return (bl2i and b_lin_bounded and bLipschitz), np.array(gammas)

    def compute_L2_taylor(self, u_eq, x_eq, epsilon=1e-7, solver="MOSEK"):
        inputs = (u_eq, x_eq)
        jac = torch.autograd.functional.jacobian(self.forward, inputs)
        B, A = jac[0]
        D, C = jac[1]
        with torch.no_grad():
            A = A.squeeze(0).squeeze(1).detach().numpy()
            B = B.squeeze(0).squeeze(1).detach().numpy()
            C = C.squeeze(0).squeeze(1).detach().numpy()
            D = D.squeeze(0).squeeze(1).detach().numpy()
            nx = A.shape[0]
            nu = B.shape[1]
            ny = C.shape[0]
            P = Variable((nx, nx), "P", PSD=True)
            gam = Variable()
            M = bmat(
                [
                    [A.T @ P + P @ A, P @ B, C.T],
                    [B.T @ P, -gam * np.eye(nu), D.T],
                    [C, D, -gam * np.eye(ny)],
                ]
            )
            constraints = [
                M << -epsilon * np.eye(nx + nu + ny),
                P - (epsilon) * np.eye(nx) >> 0,
                gam - epsilon >= 0,
            ]
            objective = Minimize(gam)  # Feasibility problem

            prob = Problem(objective, constraints=constraints)
            prob.solve(solver)
            if prob.status not in ["infeasible", "unbounded"]:
                gamma = np.sqrt(gam.value)
            else:
                gamma = np.Inf
        return gamma

    def save_weights(self):
        A, B, C = self.linmod.eval_()

        weights = self.fx.extractWeightsSandwich()
        Wfx = weights[0][:, :self.nx]
        Wfu = weights[0][:, self.nx:]
        Wf = weights[1]

        mdict = {
            'Wfx': Wfx.detach().numpy(),
            'Wfu': Wfu.detach().numpy(),
            'Wf': Wf.detach().numpy(),
            'A': A.detach().numpy(),
            'B': B.detach().numpy(),
            'C': C.detach().numpy()
        }
        savemat('weights.mat', mdict)
