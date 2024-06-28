import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.linalg import cholesky
from ctrl_nmod.linalg.utils import isSDP
from geotorch_custom.parametrize import is_parametrized


class ContractingREN(nn.Module):
    """
    Contracting Recurrent Equilibrium Network (REN) implementation.

    This class represents a Contracting REN, ensuring strong nonlinear stability
    as per the specified contraction conditions in equations 21 to 26.

    Args:
        nx (int): Number of internal states.
        ny (int): Number of outputs.
        nu (int): Number of inputs.
        nq (int): Number of non-linear states.
        sigma (str): Activation function type ('tanh', 'sigmoid', 'relu', or 'identity').
        epsilon (float): Small positive scalar for numerical stability.
        device (str): Computation device ('cpu' or 'cuda').
        bias (bool, optional): Whether to use bias terms. Defaults to False.
        alpha (float, optional): Lower bound of contraction rate. Defaults to 0.0.
    """

    def __init__(self, nx: int, ny: int, nu: int, nq: int, sigma: str,
                 epsilon: float, device: str, bias: bool = False, alpha: float = 0.0, feedthrough=False) -> None:
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nq = nq
        self.epsilon = epsilon
        self.device = device
        self.alpha = alpha
        self.feedthrough = feedthrough

        # Initialization of the Weights
        self.X = Parameter(torch.randn(2 * nx + nq, 2 * nx + nq, device=device))

        self.register_buffer('I', torch.eye(2 * nx + nq, device=device))

        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D12 = Parameter(torch.randn(nq, nu, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))
        if self.feedthrough:
            self.D22 = Parameter(torch.randn(ny, ny, device=device))

        if bias:
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.register_buffer('bx', torch.zeros(nx, 1, device=device))
            self.register_buffer('bv', torch.zeros(nq, 1, device=device))
            self.register_buffer('by', torch.zeros(ny, 1, device=device))

        self.Y1 = Parameter(torch.randn(nx, nx, device=device))

        # Constrained parameters
        self.register_buffer('F', torch.zeros(nx, nx, device=device))
        self.register_buffer('B1', torch.zeros(nx, nq, device=device))
        self.register_buffer('C1', torch.zeros(nq, nx, device=device))
        self.register_buffer('E', torch.zeros(nx, nx, device=device))
        self.register_buffer('D11', torch.zeros(nq, nq, device=device))
        self.register_buffer('Lambda', torch.zeros(nq, nq, device=device))

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

        if not is_parametrized(self):  # Check if X is already parameterized to be Positive Definite
            H = self.X @ self.X.T + self.epsilon * self.I
        else:
            H = self.X

        # Split H into appropriate blocks
        h1, h2, h3 = torch.split(H, [self.nx, self.nq, self.nx], dim=0)
        H11, _, H13 = torch.split(h1, [self.nx, self.nq, self.nx], dim=1)
        H21, H22, H23 = torch.split(h2, [self.nx, self.nq, self.nx], dim=1)
        H31, H32, H33 = torch.split(h3, [self.nx, self.nq, self.nx], dim=1)

        # Update buffers
        self.F = H31
        self.B1 = H32
        self.P = H33
        self.C1 = -H21

        self.E = 0.5 * (H11 + 1 / self.alpha**2 * self.P + self.Y1 - self.Y1.T)
        Phi = torch.diag(torch.diag(H22))
        L = -torch.tril(H22, -1)

        # Update Lambda and D11
        self.Lambda = 0.5 * Phi
        self.D11 = L

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Current state.
            u (torch.Tensor): Input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Next state and output.
        """
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.P)
        dx = x @ (E_inv @ self.A).T + w @ (E_inv @ self.B1).T + u @ (E_inv @ self.B2).T
        y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        return dx, y

    def _solve_w(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Solve for the internal non-linear state w.

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
            v[:, k] = (1 / self.Lambda[k, k]) * (x @ self.C1[k, :] + w @ self.D11[k, :] + u @ self.D12[k, :] + self.bv[k])
            w[:, k] = self.act(v[:, k].clone())
        return w

    def _check_contraction(self) -> torch.Tensor:
        """
        Check if the network satisfies the contraction conditions.

        Returns:
            Tensor: True if the network satisfies the conditions, False otherwise.
        """
        self._frame()  # Update parameters before checking contraction

        # Construct H(theta_cvx)
        H11 = self.E + self.E.T - 1 / self.alpha**2 * self.P
        H_upper = torch.cat([H11, -self.C1.T, self.F.T], dim=1)
        H_middle = torch.cat([-self.C1, self.W, self.B1.T], dim=1)
        H_lower = torch.cat([self.F, self.B1, self.P], dim=1)
        H_cvx = torch.cat([H_upper, H_middle, H_lower], dim=0)

        # Check if H_cvx is positive definite
        is_positive_definite = torch.all(torch.linalg.eigvals(H_cvx).eigenvalues.real > 0)
        return is_positive_definite


class _ImplicitQSRNetwork(Module):
    """
    Implicit QSR Network implementation.

    This class represents an implicit QSR (Quadratic Supply Rate) network,
    which is a type of neural network with specific stability properties.

    Args:
        nx (int): Number of internal states.
        ny (int): Number of outputs.
        nu (int): Number of inputs.
        nq (int): Number of non-linear states.
        sigma (str): Activation function type ('tanh', 'sigmoid', 'relu', or 'identity').
        epsilon (float): Small positive scalar for numerical stability.
        S (torch.Tensor): S matrix for QSR dissipation inequality.
        Q (torch.Tensor): Q matrix for QSR dissipation inequality.
        R (torch.Tensor): R matrix for QSR dissipation inequality.
        gamma (float): Contraction rate.
        device (str): Computation device ('cpu' or 'cuda').
        bias (bool, optional): Whether to use bias terms. Defaults to False.
        alpha (float, optional): Lower bound of contraction rate. Defaults to 0.0.
        feedthrough (bool, optional): Whether to use direct feedthrough. Defaults to True.
    """

    def __init__(self, nx: int, ny: int, nu: int, nq: int, sigma: str,
                 epsilon: float, S: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
                 gamma: float, device: str, bias: bool = False,
                 alpha: float = 0.0, feedthrough: bool = True) -> None:
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nq = nq
        self.s = max(nu, ny)
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.alpha = alpha
        self.feedthrough = feedthrough

        # Initialization of the Weights
        self.D12_tilde = Parameter(torch.randn(nq, nu, device=device))
        self.X3 = Parameter(torch.randn(self.s, self.s, device=device))
        self.Y3 = Parameter(torch.randn(self.s, self.s, device=device))
        self.Y1 = Parameter(torch.randn(nx, nx, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))

        if bias:
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.register_buffer('bx', torch.zeros(nx, 1, device=device))
            self.register_buffer('bv', torch.zeros(nq, 1, device=device))
            self.register_buffer('by', torch.zeros(ny, 1, device=device))

        self.X = Parameter(torch.randn(2 * nx + nq, 2 * nx + nq, device=device))

        # Constrained parameters
        self.register_buffer('F', torch.zeros(nx, nx, device=device))
        self.register_buffer('D11', torch.zeros(nq, nq, device=device))
        self.register_buffer('C1', torch.zeros(nq, nx, device=device))
        self.register_buffer('B1', torch.zeros(nx, nq, device=device))
        self.register_buffer('D12', torch.zeros(nq, nu, device=device))
        self.register_buffer('D22', torch.zeros(ny, nu, device=device))
        self.register_buffer('E', torch.zeros(nx, nx, device=device))
        self.register_buffer('Lambda', torch.zeros(nq, nq, device=device))

        self.R = R
        self.Q = Q
        self.S = S

        if self.feedthrough:
            try:
                self.Lq = cholesky(-Q)
            except torch.linalg.LinAlgError:
                self.Q = Q - self.epsilon * torch.eye(Q.shape[0])
                self.Lq = cholesky(-self.Q)
            self.Lr = cholesky(
                self.R - self.S @ torch.inverse(self.Q) @ self.S.T)

        self._frame()

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
            print("Error. Invalid sigma function. Using Tanh().")
            self.act = nn.Tanh()

    def __str__(self) -> str:
        return f"ImplicitQSRNet_alpha_{self.alpha}_gamma_{self.gamma}"

    def _frame(self) -> None:
        """
        Update the network parameters based on the current state.
        """
        M = F.linear(self.X3, self.X3) + (self.Y3 - self.Y3.T + (self.epsilon * torch.eye(self.s, device=self.device)))
        if self.feedthrough:
            M_tilde = F.linear(torch.eye(self.s, device=self.device) - M,
                               torch.inverse(torch.eye(self.s, device=self.device) + M).T)
            M_tilde = M_tilde[0:self.ny, 0:self.nu]

            self.D22 = torch.inverse(self.Q) @ self.S.T + \
                torch.inverse(self.Lq) @ M_tilde @ self.Lr

        R_capital = self.R + self.S @ self.D22 + \
            self.D22.T @ self.S.T + self.D22.T @ self.Q @ self.D22

        C2_tilde = (self.D22.T @ self.Q + self.S) @ self.C2
        D21_tilde = (self.D22.T @ self.Q) @ self.D21 - self.D12_tilde.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)
        vec_C2_D21 = torch.cat(
            [self.C2.T, self.D21.T, torch.zeros((self.nx, self.ny), device=self.device)], 0)

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

        self.F = H31
        self.B1 = H32
        self.P = H33
        self.C1 = -H21
        self.E = 0.5 * (H11 + (1 / self.alpha**2) * self.P + self.Y1 - self.Y1.T)
        self.Lambda = 0.5 * torch.diag(torch.diag(H22))

        L = -torch.tril(H22, -1)
        lambda_inv = torch.inverse(self.Lambda)
        self.D11 = lambda_inv @ L
        self.D12 = lambda_inv @ self.D12_tilde

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Current state.
            u (torch.Tensor): Input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Next state and output.
        """
        self._frame()
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.E)
        dx = x @ (E_inv @ self.F).T + w @ (E_inv @
                                           self.B1).T + u @ (E_inv @ self.B2).T
        if self.feedthrough:
            y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        else:
            y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Solve for the internal non-linear state w.

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


class _System_general(nn.Module):
    """
    General system implementation.

    This class should not be used directly, but is used by the REN class.

    Args:
        nx (int): Number of internal states.
        ny (int): Number of outputs.
        nu (int): Number of inputs.
        nq (int): Number of non-linear states.
        sigma (str): Activation function type.
        epsilon (float): Small positive scalar for numerical stability.
        device (str): Computation device.
        bias (bool, optional): Whether to use bias terms. Defaults to False.
        linear_output (bool, optional): Whether to force linear output. Defaults to False.
    """

    def __init__(self, nx: int, ny: int, nu: int, nq: int, sigma: str,
                 epsilon: float, device: str, bias: bool = False,
                 linear_output: bool = False):
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nq = nq
        self.epsilon = epsilon
        self.device = device

        # Initialization of the Weights
        self.D12_tilde = Parameter(torch.randn(nq, nu, device=device))
        self.Y1 = Parameter(torch.zeros(nx, nx, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))

        if bias:
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.register_buffer('bx', torch.zeros(nx, 1, device=device))
            self.register_buffer('bv', torch.zeros(nq, 1, device=device))
            self.register_buffer('by', torch.zeros(ny, 1, device=device))

        self.X = Parameter(torch.eye(2 * nx + nq, 2 * nx + nq, device=device))

        # Constrained parameters
        self.register_buffer('F', torch.zeros(nx, nx, device=device))
        self.register_buffer('D11', torch.zeros(nq, nq, device=device))
        self.register_buffer('C1', torch.zeros(nq, nx, device=device))
        self.register_buffer('B1', torch.zeros(nx, nq, device=device))
        self.register_buffer('D12', torch.zeros(nq, nu, device=device))
        self.register_buffer('D22', torch.zeros(ny, nu, device=device))
        self.register_buffer('E', torch.zeros(nx, nx, device=device))
        self.register_buffer('Lambda', torch.zeros(nq, nq, device=device))

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
            print("Error. Invalid sigma function. Using Tanh().")
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Current state.
            u (torch.Tensor): Input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Next state and output.
        """
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.E)
        dx = x @ (E_inv @ self.F).T + w @ (E_inv @
                                           self.B1).T + u @ (E_inv @ self.B2).T
        y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Solve for the internal non-linear state w.

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
            v[:, k] = (1/self.Lambda[k, k]) * (x @ self.C1[k, :].T +
                                               w @ self.D11[k, :].T + u @ self.D12[k, :].T + self.bv[k])
            w[:, k] = self.act(v[:, k])
        return w


class REN(nn.Module):
    """
    Recurrent Equilibrium Network (REN) base class.

    This class implements various types of RENs with different stability properties.

    Args:
        nx (int): Number of internal states.
        ny (int): Number of outputs.
        nu (int): Number of inputs.
        nq (int): Number of non-linear states.
        sigma (str): Activation function type.
        epsilon (float): Small positive scalar for numerical stability.
        mode (str): Stability property to ensure ('c', 'rl2', 'input_p', 'output_p', or 'general').
        gamma (float): L2 Lipschitz constant (for 'rl2' mode).
        device (str): Computation device.
        bias (bool): Whether to use bias terms.
        ni (float): Input passivity coefficient (for 'input_p' mode).
        rho (float): Output passivity coefficient (for 'output_p' mode).
        alpha (float): Lower bound of contraction rate.
        linear_output (bool): Whether to force linear output.
        feedthrough (bool): Whether to use direct feedthrough.
        save_path (str, optional): Custom save path for the model.
    """

    def __init__(self, nx: int = 5, ny: int = 5, nu: int = 5, nq: int = 5,
                 sigma: str = "tanh", epsilon: float = 1.0e-2, mode: str = "c",
                 gamma: float = 1., device: str = "cpu", bias: bool = False,
                 ni: float = 1., rho: float = 1., alpha: float = 0.0,
                 linear_output: bool = False, feedthrough: bool = True,
                 save_path: str = None):
        super().__init__()

        self.mode = mode.lower()
        self.sigma = sigma
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.bias = bias
        self.ni = ni
        self.rho = rho
        self.alpha = alpha
        self.linear_output = linear_output
        self.feedthrough = feedthrough

        self.nx, self.ny, self.nu, self.nq = nx, ny, nu, nq

        if save_path is None:
            self.save_path = os.path.join(
                os.getcwd(), f'model_{self.mode}.pkl')
        else:
            self.save_path = save_path

        if self.mode == "general":
            self.sys = _System_general(nx, ny, nu, nq, sigma, epsilon, device,
                                       bias=bias, linear_output=linear_output)
        else:  # QSR
            if self.mode == "rl2":
                Q = -(1./gamma) * torch.eye(ny, device=device)
                R = gamma * torch.eye(nu, device=device)
                S = torch.zeros(nu, ny, device=device)
            elif self.mode == "input_p":
                assert nu == ny, "Input and Output dimensions must be equal"
                Q = torch.zeros(ny, device=device)
                R = -2 * ni * torch.eye(nu, device=device)
                S = torch.eye(nu, ny, device=device)
            elif self.mode == "output_p":
                assert nu == ny, "Input and Output dimensions must be equal"
                Q = -2 * rho * torch.eye(nu, device=device)
                R = torch.zeros(nu, device=device)
                S = torch.eye(nu, ny, device=device)
            else:
                raise ValueError(
                    "Invalid mode. Please use 'c', 'rl2', 'input_p' or 'output_p'.")

            self.sys = _ImplicitQSRNetwork(nx, ny, nu, nq, sigma, epsilon,
                                           S=S, Q=Q, R=R, gamma=gamma, device=device,
                                           bias=bias, alpha=alpha, feedthrough=feedthrough)

    def forward(self, u: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the REN.

        Args:
            u (torch.Tensor): Input.
            x (torch.Tensor): Current state.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Next state and output.
        """
        return self.sys(x, u)

    def get_obs_size(self) -> int:
        """
        Get the size of the observation space.

        Returns:
            int: Size of the observation space.
        """
        return self.nb

    def clone(self):
        """
        Create a deep copy of the current REN.

        Returns:
            REN: A new instance with the same parameters and state.
        """
        copy = type(self)(
            self.nx, self.ny, self.nu, self.nq, self.sigma, self.epsilon,
            self.mode, self.gamma, self.device, self.bias, self.ni, self.rho,
            self.alpha, self.linear_output, self.feedthrough, self.save_path
        )
        copy.load_state_dict(self.state_dict())
        return copy
