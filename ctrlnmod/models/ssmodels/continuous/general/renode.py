import torch
from torch import nn
from torch.nn.parameter import Parameter
from ctrlnmod.linalg.utils import is_positive_definite, sqrtm
from geotorch_custom.parametrize import is_parametrized
import geotorch_custom as geo
from typing import Optional, Tuple
from ctrlnmod.lmis import AbsoluteStableLFT
from ctrlnmod.models.ssmodels.base import SSModel
from typing import Literal


class RENODE(SSModel):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nq: int,
        sigma: str = 'relu',
        bias: bool = False,
        feedthrough: bool = False,
        out_eq_nl: bool = False,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super(RENODE, self).__init__(nu, ny, nx)


        self.nq = nq
        self.s = max(nu, ny)
        self.device = device
        self.feedthrough = feedthrough
        self.out_eq_nl = out_eq_nl
        self.bias = bias

        # Initialization of the Weights
        self.D12 = Parameter(torch.randn(nq, nu, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        
        if out_eq_nl:
            self.D21 = Parameter(torch.randn(ny, nq, device=device))
        else:   
            self.register_buffer('D21', torch.zeros((ny, nq), device=device))
        
        self.A = Parameter(torch.zeros(nx, nx, device=device))
        self.D11 = Parameter(torch.zeros(nq, nq, device=device))
        self.C1 = Parameter(torch.zeros(nq, nx, device=device))
        self.B1 = Parameter(torch.zeros(nx, nq, device=device))

        if self.feedthrough:
            self.D22 = Parameter(torch.zeros(ny, nu, device=device))
        else:
            self.register_buffer('D22', torch.zeros(ny, nu, device=device))
        
        if bias:
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.register_buffer('bx', torch.zeros(nx, 1, device=device))
            self.register_buffer('bv', torch.zeros(nq, 1, device=device))
            self.register_buffer('by', torch.zeros(ny, 1, device=device))

        self.sigma = sigma
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

    def _frame(self) -> Tuple[torch.Tensor, ...]:
        return (self.A, self.B1, self.B2, self.C1, self.C2, 
                self.D11, self.D12, self.D21, self.D22)

    def forward(self, u: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A, B1, B2, C1, C2, D11, D12, D21, D22 = self._frame()
        w = self._solve_w(x, u, C1, D11, D12)
        dx = x @ A.T + w @ B1.T + u @ B2.T + self.bx.T
        if self.feedthrough:
            y = x @ C2.T + w @ D21.T + u @ D22.T + self.by.T
        else:
            y = x @ C2.T + w @ D21.T + self.by.T
        return dx, y

    def _solve_w(self, x: torch.Tensor, u: torch.Tensor, C1: torch.Tensor, 
                 D11: torch.Tensor, D12: torch.Tensor) -> torch.Tensor:
        nb = x.shape[0]
        w = torch.zeros(nb, self.nq, device=self.device)
        v = torch.zeros(nb, self.nq, device=self.device)

        for k in range(self.nq):
            v[:, k] = x @ C1[k, :] + w.clone() @ D11[k, :] + u @ D12[k, :] + self.bv[k]
            w[:, k] = self.act(v[:, k])
        return w

    def check(self) -> Tuple[bool, dict]:
        return True, {}

    def _right_inverse(self):
        pass

    def init_weights_(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,  **kwargs) -> None:
        with torch.no_grad():
            self.A.copy_(A)
            self.B2.copy_(B)
            self.C2.copy_(C)
            self.D11.copy_(1e-6 * torch.tril(torch.randn_like(self.D11), -1))
            self.B1.copy_(torch.zeros_like(self.B1))
            self.C1.copy_(torch.randn_like(self.C1))
            self.D21.copy_(torch.zeros_like(self.D21))
            if self.bias:
                self.bx.copy_(torch.zeros_like(self.bx))
                self.bv.copy_(torch.randn_like(self.bv))
                self.by.copy_(torch.zeros_like(self.by))

    def clone(self):
        copy = type(self)(
            nu=self.nu,
            nx=self.nx,
            ny=self.ny,
            nq=self.nq,
            sigma=self.sigma,
            bias=self.bias,
            feedthrough=self.feedthrough,
            out_eq_nl=self.out_eq_nl,
            device=self.device
        )
        copy.load_state_dict(self.state_dict())
        return copy



class ContractingRENODE(RENODE):
    type Parameterization = Literal['square', 'expm']
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nq: int,
        sigma: str = 'relu',
        alpha: float = 0.0,
        epsilon: float = 1e-4,
        bias: bool = False,
        feedthrough: bool = False,
        out_eq_nl: bool = False,
        param: Parameterization = 'square',
        device: torch.device = torch.device('cpu')
    ) -> None:
        super(ContractingRENODE, self).__init__(
            nu, ny, nx, nq, sigma, bias, feedthrough, out_eq_nl, device
        )

        self.param = param
        self.alpha = torch.tensor([alpha], device=device, requires_grad=False)
        self.epsilon = epsilon

        self.P_inv = Parameter(torch.eye(nx, nx, device=device), requires_grad=True)
        self.S = Parameter(torch.zeros(nx, nx, device=device), requires_grad=True)
        self.X = Parameter(torch.eye(nx + nq, nx + nq, device=device), requires_grad=True)
        self.X_test = Parameter(torch.eye(nx + nq, nx + nq, device=device), requires_grad=True)
        self.U = Parameter(torch.zeros(nx, nq, device=device), requires_grad=True)

        if param != 'square':
            geo.positive_definite(self, 'X', triv=param)
            geo.positive_definite(self, 'P_inv', triv=param)
            geo.skew_symmetric(self, 'S')

        self.register_buffer('I', torch.eye(nx + nq, device=device))
        self.register_buffer('Ix', torch.eye(nx, device=device))
        self.register_buffer('Lambda', torch.zeros((nq, nq), device=device))

    def _right_inverse(self, A: torch.Tensor, B1: torch.Tensor, C1: torch.Tensor, 
                       D11: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        M, Lambda, P = AbsoluteStableLFT.solve(A, B1, C1, D11, alpha, tol=1e-4)
        P_inv = torch.inverse(P)
        Q = M[:self.nx, :self.nx]
        S = 0.5 * Q + P @ (A + alpha * self.Ix)
        U = C1.T @ Lambda
        return P_inv, M, S, U, Lambda

    def init_weights_(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, **kwargs) -> None:
        super().init_weights_(A, B, C)
        with torch.no_grad():

            self.alpha = torch.tensor([1e-3], device=self.device, requires_grad=False)
            P_inv, X, S, U, Lambda = self._right_inverse(self.A, self.B1, self.C1, self.D11, self.alpha)

            self.U.copy_(U)

            if not is_parametrized(self):
                self.P_inv.copy_(sqrtm(P_inv))
                self.X.copy_(sqrtm(X))
                self.S.copy_(torch.tril(S, -1))
            else:
                self.P_inv = P_inv
                self.X = X
                self.S = S

    def _frame(self) -> Tuple[torch.Tensor, ...]:
        if not is_parametrized(self):
            H = self.X @ self.X.T + self.epsilon * self.I
            S = self.S - self.S.T
            P_inv = (self.P_inv @ self.P_inv.T).clone()
        else:
            H = self.X
            S = self.S
            P_inv = self.P_inv

        h1, h2 = torch.split(H, [self.nx, self.nq], dim=0)
        H11, H12 = torch.split(h1, [self.nx, self.nq], dim=1)
        H21, H22 = torch.split(h2, [self.nx, self.nq], dim=1)

        A = P_inv @ (-0.5 * H11 + S) - self.alpha * self.Ix
        B1 = P_inv @ (-H12 - self.U)
        Lambda = 0.5 * torch.diag(torch.diag(H22))
        self.Lambda = Lambda.detach().clone()
        L = -torch.tril(H22, -1)
        Lambda_inv = torch.inverse(Lambda)
        D11 = Lambda_inv @ L
        C1 = Lambda_inv @ self.U.T

        return (A, B1, self.B2, C1, self.C2, D11, self.D12, self.D21, self.D22)
    
    def check(self) -> Tuple[bool, dict]:
        A, B1, _, C1, _, D11, _, _, _ = self._frame()

        if not is_parametrized(self):
            P_inv = (self.P_inv @ self.P_inv.T).clone()
        else:
            P_inv = self.P_inv

        P = torch.inverse(P_inv)
        H11 = -(A.T @ P + P @ A + 2 * self.alpha * P)
        H12 = -(C1.T @ torch.diag(torch.diag(D11)) + P @ B1)
        H22 = 2* self.Lambda -self.Lambda @ D11 - D11.T @ self.Lambda

        H_upper = torch.cat([H11, H12], dim=1)
        H_lower = torch.cat([H12.T, H22], dim=1)
        H_cvx = torch.cat([H_upper, H_lower], dim=0)

        return is_positive_definite(H_cvx), {}

    def __str__(self) -> str:
        return f'ContractingRENODE_{self.alpha.item()}'
    
    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the ContractingRENODE model.
        Includes all important configuration parameters and model state information.
        """
        # Basic dimensions
        dims = f"nu={self.nu}, ny={self.ny}, nx={self.nx}, nq={self.nq}"
        
        # Configuration flags
        config_flags = []
        if self.bias:
            config_flags.append("bias=True")
        if self.feedthrough:
            config_flags.append("feedthrough=True")
        if self.out_eq_nl:
            config_flags.append("out_eq_nl=True")
        
        config_str = ", ".join(config_flags) if config_flags else "no optional features"
        
        # Parameterization info
        param_info = f"param='{self.param}'"
        
        # Stability parameters
        alpha_val = self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha
        stability_params = f"alpha={alpha_val:.6f}, epsilon={self.epsilon:.1e}"
    
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_count = f"{trainable_params}/{total_params} trainable params"
        
        # Construct the full representation
        repr_str = (
            f"ContractingRENODE(\n"
            f"  dimensions: {dims}\n"
            f"  activation: {self.sigma}\n"
            f"  stability: {stability_params}\n"
            f"  parametrization: {param_info}\n"
            f"  configuration: {config_str}\n"
            f"  parameters: {param_count}\n"
            f")"
        )
        
        return repr_str

    def clone(self):
        new_model = ContractingRENODE(
            nu=self.nu,
            nx=self.nx,
            ny=self.ny,
            nq=self.nq,
            sigma=self.sigma,
            alpha=self.alpha.item(),
            epsilon=self.epsilon,
            bias=self.bias,
            feedthrough=self.feedthrough,
            out_eq_nl=self.out_eq_nl,
            param=self.param,
            device=self.device
        )

        new_model.load_state_dict(self.state_dict())

        for param_name, param in self.named_parameters():
            new_model.get_parameter(param_name).requires_grad_(param.requires_grad)

        return new_model
    

class DissipativeRENODE(ContractingRENODE):
    """
    This is a dissipative REN. It is a contractive REN satisfying
    a dissipation inequality given by the matrices QSR.
    """

    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nq: int,
        sigma: str,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        alpha: float = 0.0,
        epsilon: float = 1e-4,
        bias: bool = False,
        feedthrough: bool = True,
        out_eq_nl: bool = False,
        param: Optional[str] = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super(DissipativeRENODE, self).__init__(
            nu, ny, nx, nq, sigma, alpha, epsilon, bias, feedthrough, out_eq_nl, param, device
        )

        self.Q = Q.to(device)
        self.S = S.to(device)
        self.R = R.to(device)

        if self.feedthrough:
            if param is not None:  # We parameterize directly N
                self.N = Parameter(torch.randn(ny, nu, device=device))
                geo.orthogonal(self, 'N', triv=param)
            else:
                # New variables for D22 calculation
                self.X3 = Parameter(torch.randn(max(nu, ny), max(nu, ny), device=device))
                self.Y3 = Parameter(torch.randn(max(nu, ny), max(nu, ny), device=device))
                self.Z3 = Parameter(torch.randn(abs(nu - ny), min(nu, ny), device=device))

            # Ensure Q is negative definite
            if not torch.all(torch.linalg.eigvals(Q).real < 0):
                self.Q = Q - epsilon * torch.eye(Q.shape[0], device=device)

            try:
                self.Lq = torch.linalg.cholesky(-self.Q)
            except torch.linalg.LinAlgError:
                raise ValueError("Q is not negative definite even after adjustment")

            # Calculate LR
            R_tilde = R - S @ torch.inverse(self.Q) @ S.T
            if not torch.all(torch.linalg.eigvals(R_tilde).real > 0):
                raise ValueError("R - SQ^(-1)S^T is not positive definite")
            self.Lr = torch.linalg.cholesky(R_tilde)

    def _compute_N(self) -> torch.Tensor:
        if self.param is None:  # Default is rectangular cayley
            M = (self.X3 @ self.X3.T + self.Y3 - self.Y3.T + self.Z3 @ self.Z3.T + 
                 self.epsilon * torch.eye(max(self.nu, self.ny), device=self.device))

            if self.ny >= self.nu:
                N_upper = (torch.eye(self.nu, device=self.device) - M) @ torch.inverse(torch.eye(self.nu, device=self.device) + M)
                N_lower = -2 * self.Z3 @ torch.inverse(torch.eye(self.nu, device=self.device) + M)
                N = torch.cat([N_upper, N_lower], dim=0)
            else:
                N = torch.cat([
                    torch.inverse(torch.eye(self.ny, device=self.device) + M) @ (torch.eye(self.ny, device=self.device) - M),
                    -2 * torch.inverse(torch.eye(self.ny, device=self.device) + M) @ self.Z3.T
                ], dim=1)
        else:
            N = self.N
        return N

    def _frame(self) -> Tuple[torch.Tensor, ...]:
        """
        Update the network parameters based on the current state.
        """
        if self.feedthrough:
            N = self._compute_N()
            # Calculate D22
            D22 = -torch.inverse(self.Q) @ self.S.T + torch.inverse(self.Lq) @ N @ self.Lr
        else:
            D22 = torch.zeros(self.ny, self.nu, device=self.device)

        R_capital = self.R + self.S @ D22 + D22.T @ self.S.T + D22.T @ self.Q @ D22

        C2_tilde = (D22.T @ self.Q + self.S) @ self.C2
        D21_tilde = (D22.T @ self.Q) @ self.D21 - self.D12.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)
        vec_C2_D21 = torch.cat([self.C2.T, self.D21.T, torch.zeros((self.nx, self.ny), device=self.device)], 0)

        Hs1 = self.X @ self.X.T
        Hs2 = self.epsilon * torch.eye(2 * self.nx + self.nq, device=self.device)
        H_cvx = Hs1 + Hs2

        Hs3 = V_tilde @ torch.linalg.solve(R_capital, V_tilde.T)
        Hs4 = vec_C2_D21 @ self.Q @ vec_C2_D21.T

        H = H_cvx + Hs3 - Hs4

        h1, h2, h3 = torch.split(H, [self.nx, self.nq, self.nx], dim=0)
        H11, H12, H13 = torch.split(h1, [self.nx, self.nq, self.nx], dim=1)
        H21, H22, H23 = torch.split(h2, [self.nx, self.nq, self.nx], dim=1)
        H31, H32, H33 = torch.split(h3, [self.nx, self.nq, self.nx], dim=1)

        F = H31
        B1 = H32
        P = H33
        C1 = -H21
        E = 0.5 * (H11 + (1 / self.alpha**2) * P + self.S - self.S.T)
        Lambda = 0.5 * torch.diag(torch.diag(H22))

        L = -torch.tril(H22, -1)
        D11 = L

        A = torch.inverse(P) @ (-E - E.T + (1 / self.alpha**2) * P)
        B2 = self.B2
        C2 = self.C2
        D12 = self.D12
        D21 = self.D21

        return A, B1, B2, C1, C2, D11, D12, D21, D22

    def check(self) -> Tuple[bool, dict]:
        """
        Check if the network satisfies the QSR dissipation inequality.

        Returns:
            bool: True if the network satisfies the inequality, False otherwise.
        """
        A, B1, B2, C1, C2, D11, D12, D21, D22 = self._frame()
        P = torch.inverse(self.P_inv)
        H11 = A.T @ P + P @ A + C2.T @ self.Q @ C2
        H12 = P @ B1 + C2.T @ self.Q @ D21
        H13 = P @ B2 + C2.T @ self.Q @ D22 + C2.T @ self.S
        H21 = H12.T
        H22 = 2 * torch.diag(torch.diag(D11)) - D11 - D11.T + D21.T @ self.Q @ D21
        H23 = D21.T @ self.Q @ D22 + D21.T @ self.S - D12.T
        H31 = H13.T
        H32 = H23.T
        H33 = self.R + D22.T @ self.Q @ D22 + self.S @ D22 + D22.T @ self.S.T

        H = torch.cat([
            torch.cat([H11, H12, H13], dim=1),
            torch.cat([H21, H22, H23], dim=1),
            torch.cat([H31, H32, H33], dim=1)
        ], dim=0)

        return is_positive_definite(H, tol=1e-6), {}

    def __str__(self) -> str:
        return f'DissipativeRENODE_{self.alpha.item()}'

    def clone(self):
        new_model = DissipativeRENODE(
            nu=self.nu,
            nx=self.nx,
            ny=self.ny,
            nq=self.nq,
            sigma=self.sigma,
            Q=self.Q.clone(),
            S=self.S.clone(),
            R=self.R.clone(),
            alpha=self.alpha.item(),
            epsilon=self.epsilon,
            bias=self.bias,
            feedthrough=self.feedthrough,
            out_eq_nl=self.out_eq_nl,
            param=self.param,
            device=self.device
        )

        new_model.load_state_dict(self.state_dict())

        for param_name, param in self.named_parameters():
            new_model.get_parameter(param_name).requires_grad_(param.requires_grad)

        return new_model