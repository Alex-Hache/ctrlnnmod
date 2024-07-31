import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from ctrl_nmod.linalg.utils import sqrtm
import numpy as np
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.atoms.affine.bmat import bmat


class L2BoundedLinear(Module):
    """
        Create a linear continuous-time state-space model with a prescribed L2 gain and alpha stability.

        attributes
        ----------

            * nu : int
                input dimension
            * ny : int
                output dimension
            * nx : int
                state dimension
            * gamma : Tensor
                precribed H2 norm
            * alpha : Tensor
                inverse of observability grammian
            * Q : Tensor
                Positive definite matrix
            * S : Tensor
                skew-symmetric matrix
            * G : Tensor
                Output matrix
            * H : Tensor
                Semi-orthogonal matrix
    """
    def __init__(self, nu: int, ny: int, nx: int, gamma: float, alpha: float = 0.0,
                 scaleH=1.0, epsilon=0.0) -> None:
        super(L2BoundedLinear, self).__init__()
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.gamma = Tensor([gamma])
        self.alpha = Tensor([alpha])
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
        geo.skew_symmetric(self, 'S')
        geo.orthogonal(self, 'H')  # TODO relax constraint taking alpha * P into account

    def __repr__(self):
        return "Hinf_Linear_ss" + f"_alpha_{self.alpha}" + f"_gamma_{self.gamma}"

    def forward(self, u, x):
        A, B, C = self.frame()

        dx = x @ A.T + u @ B.T
        y = x @ C.T
        return dx, y

    def frame(self):
        A = (-0.5 * (self.Q + self.G.T @ self.G + self.eps * self.Ix) + self.S) @ self.P - self.alpha * self.Ix
        B = self.gamma * sqrtm(self.Q) @ (self.scaleH * self.H)  # type: ignore
        C = self.G @ self.P
        return A, B, C

    def right_inverse_(self, A, B, C, gamma: float, alpha):
        Q, P_torch, S, G, H, alph, _ = self.submersion_inv(A, B, C, float(gamma), alpha)
        self.P = P_torch
        self.Q = Q
        self.S = S
        self.G = Parameter(G)
        self.H = H
        self.alpha = alph

    def submersion_inv(self, A, B, C, gamma: float, alpha=0.0, epsilon=1e-8, solver="MOSEK", check=False):
        """
            Function from weights space to parameter space.

        """
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
                A.T @ P + P @ A + 2 * alpha * P << -(epsilon * np.eye(nx)),  # type: ignore
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
            H = H / torch.sqrt(torch.linalg.norm(H @ H.T, 2))  # We project onto Stiefeld Manifold
            alph = Tensor([alpha])
        return Q, P_torch, S, G, H, alph, gmma_lmi

    @classmethod
    def copy(cls, model):

        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            float(model.gamma),
            float(model.alpha)
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return L2BoundedLinear.copy(self)

    def check_(self):
        W = self.frame()
        try:
            _, _, _, _, _, _, gamma = self.submersion_inv(*W, float(self.gamma), check=True)
            return True, gamma
        except ValueError:
            return False, np.inf
