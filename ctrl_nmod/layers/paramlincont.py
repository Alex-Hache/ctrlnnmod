"""
    This module implements several parameterizations for continuous neural state-space models
    it includes :
        - Alpha stable linear models
        - L2 gain bounded linear models -- H inifinty norm
        - QSR dissipative linear models including :
            - L2 gain bounded
        - H2 gain
        - Incrementally QSR dissipative models including :
            - Incremental L2 gain bounded
            - Incrementally input passive
            - Incrementally output passive
"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from ctrl_nmod.linalg.utils import sqrtm
from geotorch_custom.parametrize import is_parametrized
import numpy as np
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.trace import trace


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
            return "Param_Linear_ss" + f"_{self.alpha}"
        else:
            return "Linear_ss"

    def forward(self, u, x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx, y

    def eval_(self):
        return self.A.weight, self.B.weight, self.C.weight

    def right_inverse_(self, A0, B0, C0, requires_grad=True):
        if is_parametrized(self.A):
            self.A.weight = A0
        else:
            self.A.weight = Parameter(A0)
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)
        if not requires_grad:
            if is_parametrized(self.A):
                for parameters in self.A.parameters():
                    parameters.requires_grad_(False)
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

    def init_model_(self, A0, B0, C0, is_grad=True):
        self.A.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)


class L2BoundedLinear(Module):
    def __init__(self, nu: int, ny: int, nx: int, gamma: float, alpha: float = 0.0,
                 scaleH=1.0, epsilon=0.0) -> None:
        super(L2BoundedLinear, self).__init__()
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.gamma = gamma
        self.alpha = alpha
        self.Ix = torch.eye(nx)
        self.scaleH = scaleH
        self.eps = epsilon

        self.Q = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.P = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.G = Parameter(nn.init.xavier_normal_(torch.empty(self.ny, self.nx)))
        self.H = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nu)))

        # Register relevant manifolds
        geo.positive_definite(self, 'P')
        geo.positive_definite(self, 'Q')
        geo.skew(self, 'S')
        geo.orthogonal(self, 'H')  # TODO relax constraint taking alpha * P into account

    def __repr__(self):
        return "Hinf_Linear_ss" + f"_alpha_{self.alpha}" + f"_gamma_{self.gamma}"

    def forward(self, u, x):
        A, B, C = self.eval_()

        dx = x @ A.T + u @ B.T
        y = x @ C.T
        return dx, y

    def eval_(self):
        A = (-0.5*(self.Q + self.G.T @ self.G + self.eps*self.Ix) + self.S) @ self.P - self.alpha*self.Ix
        B = self.gamma*sqrtm(self.Q) @ (self.scaleH*self.H)
        C = self.G @ self.P
        return A, B, C

    def right_inverse_(self, A, B, C, gamma, alpha):
        Q, P_torch, S, G, H, alph, _ = self.submersion_inv(A, B, C, gamma, alpha)
        self.P = P_torch
        self.Q = Q
        self.S = S
        self.G = Parameter(G)
        self.H = H
        self.alpha = alph

    def submersion_inv(self, A, B, C, gamma, alpha=0.0, epsilon=1e-8, solver="MOSEK", check=False):

        with torch.no_grad():
            A = A.detach().numpy()
            B = B.detach().numpy()
            C = C.detach().numpy()
            nx = A.shape[0]
            nu = B.shape[1]
            ny = C.shape[0]

            D = np.zeros((ny, nu))
            P = Variable((nx, nx), "P", PSD=True)
            gam = Variable()
            M = bmat(
                [
                    [A.T @ P + P @ A, P @ B, C.T],
                    [B.T @ P, -gam * np.eye(nu), D.T],  # type: ignore
                    [C, D, -gam * np.eye(ny)],  # type: ignore
                ]
            )
            constraints = [
                M << -epsilon * np.eye(nx + nu + ny),  # type: ignore
                P - (epsilon) * np.eye(nx) >> 0,  # type: ignore
                A.T @ P + P @ A + 2*alpha*P << -(epsilon*np.eye(nx)),  # type: ignore
                gam - epsilon >= 0,  # type: ignore
            ]
            objective = Minimize(gam)  # Feasibility problem

            prob = Problem(objective, constraints=constraints)
            prob.solve(solver)
            if prob.status not in ["infeasible", "unbounded"]:
                gmma_lmi = gam.value
                if gmma_lmi > gamma and check:
                    raise ValueError(f"Infeasible problem with prescribed gamma : {gamma} min value = {gmma_lmi}")
                else:
                    if gmma_lmi > gamma:
                        print(
                            "Not in manifold with gamma = {} \n New gamma value assigned : g = {} -- alpha = {}".format(
                                gamma, gmma_lmi, alpha
                            )
                        )
                        self.gamma = float(gmma_lmi)  # Assign lowest gamma found if it's higher than the one prescribed
                    else:
                        print(f"Currrent gamma value : {gmma_lmi}")
                        self.gamma = gamma
            else:
                raise ValueError("SDP problem is infeasible or unbounded")

            # Now initialize
            P_torch = Tensor(P.value)
            Q = Tensor(-M.value[:nx, :nx])  # type: ignore
            S = Tensor(P.value) @ Tensor(A) + 0.5 * Q
            G = Tensor(C) @ torch.inverse(P_torch)
            H = Tensor(1 / self.gamma * (sqrtm(Q) @ B))
            H = H/torch.sqrt(torch.linalg.norm(H@H.T, 2))  # We project onto Stiefeld Manifold
            alph = Tensor([alpha])
        return Q, P_torch, S, G, H, alph, gmma_lmi

    def check_(self):
        W = self.eval_()
        try:
            _, _, _, _, _, _, gamma = self.submersion_inv(*W, self.gamma, check=True)
            return True, gamma
        except ValueError:
            return False, np.inf


class H2BoundedLinear(Module):
    def __init__(self, nu: int, ny: int, nx: int, gamma_2: float) -> None:
        super(H2BoundedLinear, self).__init__()
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.gamma2 = Tensor([gamma_2])
        # self.alpha = alpha # TODO

        self.Wo_inv = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.M = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.C = Parameter(nn.init.xavier_normal_(torch.empty(self.ny, self.nx)))

        # Register relevant manifolds
        geo.positive_definite(self, 'Wo_inv')
        geo.skew(self, 'S')
        geo.positive_semidefinite_fixed_rank_fixed_trace(self, 'M', self.gamma2**2, self.nu)

    def __repr__(self):
        return "H2_Linear_ss" + f"_gamma2_{self.gamma2}"

    def forward(self, u, x):
        A, B, C = self.eval_()

        dx = x @ A.T + u @ B.T
        y = x @ C.T
        return dx, y

    def frame(self, tensor_name: str, tol=1e-6):
        # Assuming M is parametrized
        M = getattr(self, tensor_name)
        L, Q = torch.linalg.eigh(M)
        L_corr = torch.where(L < tol, tol, L)

        Wo_sqrt_inv = sqrtm(self.Wo_inv)
        # Wo_sqrt = sqrtm_inv(self.Wo_inv.unsqueeze(0)).squeeze(0)

        B_full = Wo_sqrt_inv @ Q @ torch.diag_embed(torch.sqrt(L_corr))
        # bbt = B_full @ B_full.T
        # 1 st check BBt = Wo -1/2 M Wo -1/2
        # print(torch.dist(bbt, Wo_sqrt_inv @ M @ Wo_sqrt_inv))

        # 2nd check
        # print(torch.dist(Wo_sqrt @ bbt @ Wo_sqrt, M))
        # print(bbt)
        return B_full[:, -self.nu:]

    def eval_(self):
        Q = self.C.T @ self.C
        A = self.Wo_inv @ (-0.5*Q + self.S)
        B = self.frame('M')
        C = self.C
        return A, B, C

    def check_(self, epsilon=1e-6):
        Wo = torch.inverse(self.Wo_inv)
        A, B, C = self.eval_()

        lyap = A.T @ Wo + Wo @ A
        dLyap = torch.dist(lyap, -C.T @ C)
        print(f"Distance to Lyapunov equation : {dLyap}")

        gamma_gram = torch.sqrt(torch.trace(B.T @ Wo @ B))
        dTrace = torch.dist(gamma_gram, self.gamma2)
        print(f"Traces : gram = {gamma_gram}  gamma2 : {self.gamma2} -- dist = {dTrace}")

        return (dLyap < epsilon) and (dTrace < epsilon), gamma_gram

    def right_inverse_(self, A, B, C, gamma2, check=False):
        Wo_inv, S, M, C = self.submersion_inv(A, B, C, gamma2=gamma2, check=check)
        self.Wo_inv = Wo_inv
        self.S = S
        self.M = M
        self.C = Parameter(C)

        # A_eval, B_eval, C_eval = self.eval_()

    def submersion_inv(self, A, B, C, gamma2=None, epsilon=1e-7, solver="MOSEK", check=False):
        with torch.no_grad():
            A = A.detach().numpy()
            B = B.detach().numpy()
            C = C.detach().numpy()
            nx = A.shape[0]
            if gamma2 is None:
                gamma2 = 0.0
            P = Variable((nx, nx), "P", PSD=True)
            Y = Variable((nx, nx), "Y", PSD=True)
            M = A.T @ P + P @ A

            constraints = [(M + C.T @ C) == -Y, P - (epsilon) * np.eye(nx) >> 0]  # type: ignore
            objective = Minimize(trace(Y))

            prob = Problem(objective, constraints=constraints)
            prob.solve(solver)

            if prob.status not in ["infeasible", "unbounded"]:
                gamma2_lmi = np.sqrt(np.trace(B.T @ P.value @ B))
                if gamma2_lmi > gamma2 and check:
                    raise ValueError(f"Infeasible problem with prescribed gamma : {gamma2} min value = {gamma2_lmi}")
                else:
                    if gamma2_lmi > gamma2:
                        print(
                            "Not in manifold with gamma2 = {} \n New gamma2 value assigned : g = {}".format(
                                gamma2, gamma2_lmi
                            )
                        )
                    self.gamma2 = Tensor(
                        [gamma2_lmi]
                    )  # Assign lowest gamma found if it's higher than the one prescribed
                    self.parametrizations.M[0].trace = self.gamma2**2   # type: ignore -- Reassign prescribed trace
                    self.parametrizations.M[0].f.eta = self.gamma2**2  # type: ignore
                    self.parametrizations.M[0].inv.eta = self.gamma2**2  # type: ignore

            else:
                raise ValueError("SDP problem is infeasible or unbounded")

            # Now initialize
            Wo = Tensor(P.value)
            Wo_inv = torch.inverse(Wo)
            Q = Tensor(-M.value[:nx, :nx])
            S = Wo @ Tensor(A) + 0.5 * Q
            Wo_sqrt = sqrtm(Wo)
            M = Tensor(Wo_sqrt @ B @ B.T @ Wo_sqrt)

        return Wo_inv, S, M, Tensor(C)
