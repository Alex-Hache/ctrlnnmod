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
from cvxpy.atoms.affine.trace import trace


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
        geo.skew_symmetric(self, 'S')
        geo.positive_semidefinite_fixed_rank_fixed_trace(self, 'M', self.gamma2**2, self.nu)

    def __repr__(self):
        return "H2_Linear_ss" + f"_gamma2_{self.gamma2}"

    def forward(self, u, x):
        A, B, C = self.frame()

        dx = x @ A.T + u @ B.T
        y = x @ C.T
        return dx, y

    def frame(self, tol=1e-6):
        # Assuming M is parametrized
        M = getattr(self, 'M')  # We access to the parameterized fixed trace and rank M
        L, Q = torch.linalg.eigh(M)
        L_corr = torch.where(L < tol, tol, L)  # Potential negative values correction

        Wo_sqrt_inv = sqrtm(self.Wo_inv)
        # Wo_sqrt = sqrtm_inv(self.Wo_inv.unsqueeze(0)).squeeze(0)

        B_full = Wo_sqrt_inv @ Q @ torch.diag_embed(torch.sqrt(L_corr))

        # bbt = B_full @ B_full.T
        # 1 st check BBt = Wo -1/2 M Wo -1/2
        # print(torch.dist(bbt, Wo_sqrt_inv @ M @ Wo_sqrt_inv))

        # 2nd check
        # print(torch.dist(Wo_sqrt @ bbt @ Wo_sqrt, M))
        # print(bbt)

        Q = self.C.T @ self.C
        A = self.Wo_inv @ (-0.5 * Q + self.S)
        C = self.C
        return A, B_full[:, -self.nu:], C

    @classmethod
    def copy(cls, model):
        '''
            This class method returns a copy of a given H2bounded model.
            We have to do this trick since self is not usable due to geotorch.
            Since when an object has parameterized attributes its class changes.
        '''
        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            float(model.gamma2)
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return H2BoundedLinear.copy(self)

    def check_(self, epsilon=1e-6):
        Wo = torch.inverse(self.Wo_inv)
        A, B, C = self.frame()

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
                        self.gamma2 = Tensor([gamma2_lmi])  # Assign lowest gamma found if it's higher than the one prescribed
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
