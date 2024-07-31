import torch
from torch import nn
from torch.nn.parameter import Parameter
from ctrl_nmod.linalg.utils import isSDP
from geotorch_custom.parametrize import is_parametrized
import geotorch_custom as geo
from typing import Optional
from ctrl_nmod.lmis import AbsoluteStableLFT
from ctrl_nmod.linalg.utils import sqrtm
"""
    This module implements Recurrent Equilibrium Networks in the acyclic case i.e.
    with no implicit layers. It is a discrete model.

"""


class RENODE(nn.Module):
    def __init__(self, nx: int, ny: int, nu: int, nq: int, sigma: str,
                 device: torch.device, bias: bool = False,
                 feedthrough: bool = True) -> None:
        super(RENODE, self).__init__()

        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nq = nq
        self.s = max(nu, ny)
        self.device = device
        self.feedthrough = feedthrough
        # Initialization of the Weights
        self.D12 = Parameter(torch.randn(nq, nu, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))

        # Potentially constrained parameters in case of C or QSR REN
        self.A = Parameter(torch.zeros(nx, nx, device=device))
        self.D11 = Parameter(torch.zeros(nq, nq, device=device))
        self.C1 = Parameter(torch.zeros(nq, nx, device=device))
        self.B1 = Parameter(torch.zeros(nx, nq, device=device))
        self.Lambda_vec = Parameter(torch.zeros(nq, device=device))
        self.register_buffer('Lambda', torch.zeros(nq, nq, device=device))

        if self.feedthrough:
            self.D22 = Parameter(torch.zeros(ny, nu, device=device))
        else:
            self.register_buffer('D22', torch.zeros(ny, nu, device=device))
        
        self.bias = bias
        
        if bias:
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.register_buffer('bx', torch.zeros(nx, 1, device=device))
            self.register_buffer('bv', torch.zeros(nq, 1, device=device))
            self.register_buffer('by', torch.zeros(ny, 1, device=device))

        self.sigma = sigma
        # Activation function
        if sigma == "tanh":
            self.act = nn.Tanh()
        elif sigma == "sigmoid":
            self.act = nn.Sigmoid()
        elif sigma == "relu":
            self.act = nn.ReLU()
        elif sigma == "identity":
            self.act = nn.Identity()
        else:
            raise ValueError("Invalid activation function specified.")

    def _frame(self):
        self.Lambda = torch.diag(self.Lambda_vec)

    def forward(self, u: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Current state.
            u (torch.Tensor): Input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Next state and output.
        """
        self._frame()
        # self.check()
        w = self._solve_w(x, u)
        dx = x @ self.A.T + w @ self.B1.T + u @ self.B2.T
        if self.feedthrough:
            y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        else:
            y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Solve for the internal non-linear state w in the acyclic case

        Args:
            x (torch.Tensor): Current state.
            u (torch.Tensor): Input.

        Returns:
            torch.Tensor: Solved internal non-linear state.
        """
        nb = x.shape[0]
        w = torch.zeros(nb, self.nq, device=self.device)
        v = torch.zeros(nb, self.nq, device=self.device)

        for k in range(self.nq):
            v[:, k] = (1 / self.Lambda[k, k]) * (x @ self.C1[k, :] + w.clone() @ self.D11[k, :] + u @ self.D12[k, :] + self.bv[k])
            w[:, k] = self.act(v[:, k].clone())
        return w

    def check(self):
        return True  # No constraints


class ContractingRENODE(RENODE):
    def __init__(self, nx: int, ny: int, nu: int, nq: int, sigma: str,
                 device: torch.device, alpha: float, epsilon: float,
                 bias: bool = False, feedthrough: bool = True,
                 param: Optional[str] = None) -> None:
        super(ContractingRENODE, self).__init__(nx, ny, nu,
                                                nq, sigma, device, bias, feedthrough)

        self.param = param
        self.alpha = alpha
        self.epsilon = epsilon

        self.P_inv = Parameter(torch.randn(nx, nx, device=device))
        self.S = Parameter(torch.randn(nx, nx, device=device))
        self.X = Parameter(torch.randn(nx + nq, nx + nq, device=device))
        self.U = Parameter(torch.randn(nx, nq, device=device))

        if param is not None:
            geo.positive_definite(self, 'X', triv=param)
            geo.positive_definite(self, 'P_inv', triv=param)
            geo.skew_symmetric(self, 'S')

        # Register identity matrix buffer
        self.register_buffer('I', torch.eye(nx + nq, device=device))
        self.register_buffer('Ix', torch.eye(nx, device=device))

        # Detach constrained parameters from tree
        self.A.requires_grad_(False)
        self.D11.requires_grad_(False)
        self.C1.requires_grad_(False)
        self.B1.requires_grad_(False)
        self.Lambda_vec.requires_grad_(False)

    def _right_inverse(self, A, B1, C1, D11, alpha):
        """
            For now only implemented for pure feedforward networks with no skip-connections.
            # TODO Implement SII initialisation for the corresponding integrator.
        """
        M, Lambda, P = AbsoluteStableLFT.solve(A, B1, C1, D11, alpha, tol=1e-2)
        P_inv = torch.inverse(P)  # type: ignore
        H11 = M[:self.nx, :self.nx]
        S = -0.5 * H11 - P @ (A + alpha * self.Ix)
        U = Parameter(C1.T @ Lambda)
        return P_inv, -M, S, U, Lambda

    def init_weights_(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        self.B2.data = B
        self.C2.data = C
        self.A.data = A

        # Initializing bias to zero
        self.bx.data = torch.zeros_like(self.bx.data)
        self.bv.data = torch.randn_like(self.bv.data)
        self.by.data = torch.zeros_like(self.by.data)

        # Initializing nonlinear part outer weights to zeros
        D11 = Parameter(torch.zeros_like(self.D11) + torch.randn_like(self.D11).tril(-1))
        B1 = Parameter(torch.zeros_like(self.B1))
        C1 = Parameter(torch.randn_like(self.C1))
        self.D21.data = torch.zeros_like(self.D21.data)
        
        alpha_A = -torch.min(torch.real(torch.linalg.eigvals(A)))
        alpha = alpha_A - 5*self.epsilon  # Set margin to alpha stability
        P_inv, X, S, U, Lambda = self._right_inverse(A, B1, C1, D11, alpha=alpha)

        self.S.data, self.U.data = S, U
        self.Lambda.data = Lambda

        if not is_parametrized(self):  # Case where PSD matrices are parameterized as square
            self.P_inv.data = sqrtm(P_inv)
            self.X.data = sqrtm(X)
        else:  # Let geotorch mechanism handle the initialization
            self.P_inv = P_inv
            self.X = X

    def _frame(self):

        # Check if X is already parameterized to be Positive Definite
        if not is_parametrized(self):
            H = self.X @ self.X.T + self.epsilon * self.I  # Positive definite
            S = self.S - self.S.T  # skew symmetric
            P_inv = (self.P_inv @ self.P_inv.T).clone()  # Positive definite
        else:
            H = self.X
            S = self.S
            P_inv = self.P_inv.clone()

        # Split H into appropriate blocks
        h1, h2 = torch.split(H, [self.nx, self.nq], dim=0)
        H11, H12 = torch.split(h1, [self.nx, self.nq], dim=1)
        H21, H22 = torch.split(h2, [self.nx, self.nq], dim=1)

        self.A.data = (P_inv @ (-0.5 * H11 - S) - self.alpha * self.Ix)
        self.B1.data = P_inv @ (-H12 - self.U)
        self.Lambda.data = 0.5 * torch.diag(torch.diag(H22))
        L = -torch.tril(H22, -1)
        Lambda_inv = torch.inverse(self.Lambda)
        # Update Lambda and D11
        self.D11.data = Lambda_inv @ L
        self.C1.data = Lambda_inv @ self.U.T

    def check(self) -> bool:
        """
        Check if the network satisfies the contraction conditions.

        Returns:
            Tensor: True if the network satisfies the conditions, False otherwise.
        """
        self._frame()  # Update parameters before checking contraction

        # Construct H(theta_cvx)
        if not is_parametrized(self):
            P_inv = (self.P_inv @ self.P_inv.T).clone()
        else:
            P_inv = self.P_inv

        P = torch.inverse(P_inv)
        H11 = -(self.A.T @ P + P @ self.A + 2 * self.alpha * P)
        H12 = -(self.C1.T @ self.Lambda + P @ self.B1)
        H22 = 2 * self.Lambda - self.Lambda @ self.D11 - self.D11.T @ self.Lambda

        H_upper = torch.cat([H11, H12], dim=1)
        H_lower = torch.cat([H12.T, H22], dim=1)

        H_cvx = torch.cat([H_upper, H_lower], dim=0)

        # Check if H_cvx is positive definite
        return isSDP(H_cvx)

    def __str__(self) -> str:
        return f'ContractinRENODE_{self.alpha}'
        
    def clone(self):
        # Create a new instance of the class
        new_model = ContractingRENODE(
            nx=self.nx,
            ny=self.ny,
            nu=self.nu,
            nq=self.nq,
            sigma=self.sigma,
            device=self.device,
            alpha=self.alpha,
            epsilon=self.epsilon,
            bias=self.bias,
            feedthrough=self.feedthrough,
            param=self.param
        )

        # Copy the state dict
        new_model.load_state_dict(self.state_dict())

        # Copy the requires_grad status of parameters
        for param_name, param in self.named_parameters():
            new_model.get_parameter(param_name).requires_grad_(param.requires_grad)

        # Copy any additional attributes that are not part of the state dict
        new_model.A.requires_grad_(self.A.requires_grad)
        new_model.D11.requires_grad_(self.D11.requires_grad)
        new_model.C1.requires_grad_(self.C1.requires_grad)
        new_model.B1.requires_grad_(self.B1.requires_grad)
        new_model.Lambda_vec.requires_grad_(self.Lambda_vec.requires_grad)

        return new_model

class DissipativeRENODE(ContractingRENODE):
    """
    This is a dissipative REN. It is a contractive REN satisfying
    a dissipation inequality given by the matrices QSR.
    """

    def __init__(self, nx: int, ny: int, nu: int, nq: int, sigma: str,
                 Q: torch.Tensor, S: torch.Tensor, R: torch.Tensor,
                 device: torch.device, alpha: float, epsilon: float,
                 bias: bool = False, feedthrough: bool = True,
                 param: Optional[str] = None) -> None:
        super(DissipativeRENODE, self).__init__(nx, ny, nu, nq,
                                                sigma, device, alpha, epsilon, bias, feedthrough, param=param)

        self.Q = Q
        self.S = S
        self.R = R

        if self.feedthrough:
            if param is not None:  # We parameterize directly N
                self.N = Parameter(torch.randn(ny, nu, device=device))
                geo.orthogonal(self, 'N', triv=param)
            else:
                # New variables for D22 calculation
                self.X3 = Parameter(torch.randn(
                    max(nu, ny), max(nu, ny), device=device))
                self.Y3 = Parameter(torch.randn(
                    max(nu, ny), max(nu, ny), device=device))
                self.Z3 = Parameter(torch.randn(
                    abs(nu - ny), min(nu, ny), device=device))

            # Ensure Q is negative definite
            if not torch.all(torch.linalg.eigvals(Q).real < 0):
                self.Q = Q - epsilon * torch.eye(Q.shape[0], device=device)

            try:
                self.Lq = torch.linalg.cholesky(-self.Q)
            except torch.linalg.LinAlgError:
                raise ValueError(
                    "Q is not negative definite even after adjustment")

            # Calculate LR
            R_tilde = R - S @ torch.inverse(self.Q) @ S.T
            if not torch.all(torch.linalg.eigvals(R_tilde).real > 0):
                raise ValueError("R - SQ^(-1)S^T is not positive definite")
            self.Lr = torch.linalg.cholesky(R_tilde)

        self.register_buffer('D22', torch.zeros(ny, nu, device=device))

    def _compute_N(self):
        if self.param is None:  # Default is rectangular cayley
            M = self.X3 @ self.X3.T + self.Y3 - self.Y3.T + self.Z3 @ self.Z3.T + \
                self.epsilon * \
                torch.eye(max(self.nu, self.ny), device=self.device)

            if self.ny >= self.nu:
                N_upper = (torch.eye(self.nu, device=self.device) - M) @ torch.inverse(torch.eye(self.nu, device=self.device) + M)
                N_lower = -2 * \
                    self.Z3 @ torch.inverse(torch.eye(self.nu,
                                            device=self.device) + M)
                N = torch.cat([N_upper, N_lower], dim=0)
            else:
                N = torch.cat([
                    torch.inverse(torch.eye(self.ny, device=self.device) + M) @ (torch.eye(self.ny, device=self.device) - M),
                    -2 * torch.inverse(torch.eye(self.ny,
                                       device=self.device) + M) @ self.Z3.T
                ], dim=1)
        else:
            N = self.N
        return N

    def _frame(self) -> None:
        """
        Update the network parameters based on the current state.
        """
        if self.feedthrough:

            # The objective of this construction is to construct a semi-orthogonal matrix N
            # such that N^TN < I our NN^T < I

            N = self._compute_N()
            # Calculate D22
            self.D22 = -torch.inverse(self.Q) @ self.S.T + \
                torch.inverse(self.Lq) @ N @ self.Lr

        R_capital = self.R + self.S @ self.D22 + \
            self.D22.T @ self.S.T + self.D22.T @ self.Q @ self.D22

        C2_tilde = (self.D22.T @ self.Q + self.S) @ self.C2
        D21_tilde = (self.D22.T @ self.Q) @ self.D21 - self.D12.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)
        vec_C2_D21 = torch.cat(
            [self.C2.T, self.D21.T, torch.zeros((self.nx, self.ny), device=self.device)], 0)

        Hs1 = self.X @ self.X.T
        Hs2 = self.epsilon * \
            torch.eye(2 * self.nx + self.nq, device=self.device)
        H_cvx = Hs1 + Hs2

        Hs3 = V_tilde @ torch.linalg.solve(R_capital, V_tilde.T)
        Hs4 = vec_C2_D21 @ self.Q @ vec_C2_D21.T

        H = H_cvx + Hs3 - Hs4

        h1, h2, h3 = torch.split(H, [self.nx, self.nq, self.nx], dim=0)
        H11, H12, H13 = torch.split(h1, [self.nx, self.nq, self.nx], dim=1)
        H21, H22, H23 = torch.split(h2, [self.nx, self.nq, self.nx], dim=1)
        H31, H32, H33 = torch.split(h3, [self.nx, self.nq, self.nx], dim=1)

        self.F = H31
        self.B1 = H32
        self.P = H33
        self.C1 = -H21
        self.E = 0.5 * (H11 + (1 / self.alpha**2) * self.P + self.Y - self.Y .T)
        self.Lambda = 0.5 * torch.diag(torch.diag(H22))

        L = -torch.tril(H22, -1)
        self.D11 = L

    def _check(self) -> bool:
        """
        Check if the network satisfies the QSR dissipation inequality.

        Returns:
            bool: True if the network satisfies the inequality, False otherwise.
        """
        self._frame()
        H11 = self.E + self.E.T - (1 / self.alpha**2) * self.P
        H21 = -self.C1
        H12 = H21.T
        H31 = self.S @ self.C2
        H13 = H31.T
        H22 = 2 * self.Lambda - self.Lambda @ self.D11 - self.D11.T @ self.Lambda
        H32 = self.S @ self.D21 - self.D12_tilde.T
        H23 = H32.T
        H33 = self.R + self.S @ self.D22 + (self.S @ self.D22).T

        H1 = torch.cat([H11, H12, H13], dim=1)
        H2 = torch.cat([H21, H22, H23], dim=1)
        H3 = torch.cat([H31, H32, H33], dim=1)

        H = torch.cat([H1, H2, H3], dim=0)

        J = torch.cat([self.F.T, self.B1.T, self.B2.T], dim=0)
        K = torch.cat([self.C2.T, self.D21.T, self.D22.T], dim=0)
        diss = -J @ torch.linalg.solve(self.P, J.T) + K @ self.Q @ K.T
        M = H + diss
        return isSDP(M, tol=1e-6)
