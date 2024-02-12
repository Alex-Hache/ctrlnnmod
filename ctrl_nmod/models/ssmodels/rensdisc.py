import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
from ctrl_nmod.linalg.utils import isSDP
from torch.nn import Module
from torch.linalg import cholesky
import os

class _ImplicitQSRNetwork(Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, S, Q, R,
                 gamma, device, bias=False, alpha=0.0, feedthrough=True) -> None:
        super().__init__()

        # Dimensions of Inputs, Outputs, States

        self.nx = nx        # no. internal-states
        self.ny = ny        # no. output
        self.nu = nu        # no. inputs
        self.nq = nq        # no. non-linear states
        self.s = np.max((nu, ny))
        self.epsilon = 0.0
        self.gamma = gamma
        self.device = device
        self.alpha = alpha
        self.feedthrough = feedthrough
        self.epsilon = epsilon

        # Initialization of the Weights:
        self.D12_tilde = Parameter(torch.randn(nq, nu, device=device))  # D12 = Lambda^{-1} DD12
        self.X3 = Parameter(torch.randn(self.s, self.s, device=device))
        self.Y3 = Parameter(torch.randn(self.s, self.s, device=device))
        self.Y1 = Parameter(torch.zeros(nx, nx, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))
        BIAS = bias
        if (BIAS):
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)

        self.X = Parameter(torch.eye(2*nx+nq, 2*nx+nq, device=device))

        # Initialization of the last Parameters which are constrained:
        self.F = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.D12 = torch.zeros(nq, nu, device=device)
        self.D22 = torch.zeros(ny, nu, device=device)
        self.E = torch.zeros(nx, nx, device=device)
        self.Lambda = torch.zeros(nq, nq, device=device)

        self.R = R
        self.Q = Q
        self.S = S

        if self.feedthrough:
            try:
                self.Lq = cholesky(Q)
            except torch.linalg.LinAlgError:
                self.Q = Q + self.epsilon * torch.eye(Q.shqpe[0])
                self.Lq = cholesky(self.Q)

            self.Lr = cholesky(self.R - self.S @ torch.inverse(self.Q) @ self.S.T)

        self._frame()  # Update from parameters to weight space
        # Choosing the activation function:
        if (sigma == "tanh"):
            self.act = nn.Tanh()
        elif (sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif (sigma == "relu"):
            self.act = nn.ReLU()
        elif (sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def _frame(self) -> None:
        M = F.linear(self.X3, self.X3) + (self.Y3 - self.Y3.T +
                                          (self.epsilon * torch.eye(self.s, device=self.device)))
        if self.feedthrough:  # Only update D22 if direct feedthrough
            M_tilde = F.linear(torch.eye(self.s, device=self.device) - M,
                               torch.inverse(torch.eye(self.s, device=self.device)+M).T)
            M_tilde = M_tilde[0:self.ny, 0:self.nu]

            self.D22 = torch.inverse(self.Q) @ self.S.T + torch.inverse(self.Lq) @ M_tilde @ self.Lr

        R_capital = self.R + self.S @ self.D22 + self.D22.T @ self.S.T + self.D22.T @ self.Q @ self.D22

        C2_tilde = (self.D22.T @ self.Q + self.S) @ self.C2
        D21_tilde = (self.D22.T @ self.Q) @ self.D21 - self.D12_tilde.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)
        vec_C2_D21 = torch.cat([self.C2.T, self.D21.T, torch.zeros((self.nx, self.ny))], 0)

        Hs1 = self.X @ self.X.T
        Hs2 = self.epsilon*torch.eye(2*self.nx+self.nq, device=self.device)
        H_cvx = Hs1 + Hs2

        Hs3 = V_tilde @ torch.linalg.solve(R_capital, V_tilde.T)
        Hs4 = vec_C2_D21 @ self.Q @ vec_C2_D21.T

        H = H_cvx + Hs3 - Hs4

        # Partition of H in --> [H1 H2;H3 H4]
        h1, h2, h3 = torch.split(H, [self.nx, self.nq, self.nx], dim=0)  # you split the matrices in three big rows
        H11, H12, H13 = torch.split(h1, [self.nx, self.nq, self.nx], dim=1)  # you split each big row in 3 chunks
        H21, H22, H23 = torch.split(h2, [self.nx, self.nq, self.nx], dim=1)
        H31, H32, H33 = torch.split(h3, [self.nx, self.nq, self.nx], dim=1)

        self.F = H31
        self.B1 = H32
        self.P = H33
        self.C1 = -H21
        self.E = 0.5*(H11 + (1/self.alpha**2)*self.P + self.Y1 - self.Y1.T)
        self.Lambda = 0.5*torch.diag(torch.diag(H22))  # Faster for matrix but not available for general tensors
        # Lambda = 0.5*torch.diag_embed(torch.diagonal(H22))  # Equivalent who is the fastest ?

        L = -torch.tril(H22, -1)
        lambda_inv = torch.inverse(self.Lambda)
        self.D11 = lambda_inv @ L
        self.D12 = lambda_inv @ self.D12_tilde

    def forward(self, x, u):

        # Then solve w_k
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.E)
        dx = x @ (E_inv @ self.F).T + w @ (E_inv @ self.B1).T + u @ (E_inv @ self.B2).T
        if self.feedthrough:
            y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        else:
            y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x, u):
        nb = x.shape[0]  # batch size
        w = torch.zeros(nb, self.nq)
        v = torch.zeros(nb, self.nq)

        # Lambda v_k = C1 x_k + D11 w_k + D12 u_k

        for k in range(0, self.nq):
            v[:, k] = (1/self.Lambda[k, k]) * (x @ self.C1[k, :] + w @ self.D11[k, :]
                                               + u @ self.D12[k, :] + self.bv[k])  # 1 dimension no need fortranspose
            w[:, k] = self.act(v[:, k])
        return w

    def _check(self) -> bool:
        self._frame()
        '''
            On construit 20 à partir des poids du réseau
        '''
        H11 = self.E + self.E.T - (1/self.alpha**2) * self.P

        H21 = -self.C1
        H12 = H21.T

        H31 = self.S @ self.C2
        H13 = H31.T

        H22 = 2*self.Lambda - self.Lambda @  self.D11 - self.D11.T @ self.Lambda

        H32 = self.S @ self.D21 - self.D12_tilde.T
        H23 = H32.T
        H33 = self.R + self.S @ self.D22 + (self.S @ self.D22).T

        H1 = torch.cat([H11, H12, H13], dim=1)
        H2 = torch.cat([H21, H22, H23], dim=1)
        H3 = torch.cat([H31, H32, H33], dim=1)

        H = torch.cat([H1, H2, H3], dim=0)

        # QSR dissipativity part
        J = torch.cat([self.F.T, self.B1.T, self.B2.T], dim=0)
        K = torch.cat([self.C2.T, self.D21.T, self.D22.T], dim=0)
        # diss = -J @ torch.inverse(self.P) @ J.T + K @ self.Q @ K.T  # Inverse of P is not symmetric
        diss = -J @ torch.linalg.solve(self.P, J.T) + K @ self.Q @ K.T  # A little bit more stable
        M = H + diss
        return isSDP(M, tol=1e-6)


class _System_general(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias=False, linear_output=False):
        """Used by the upper class NODE_REN. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -linear_output (bool, optional): choose if the output is linear,
            i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        # Dimensions of Inputs, Outputs, States

        self.nx = nx        # no. internal-states
        self.ny = ny        # no. output
        self.nu = nu        # no. inputs
        self.nq = nq        # no. non-linear states
        self.epsilon = epsilon
        self.device = device
        # Initialization of the Weights:
        self.D12_tilde = Parameter(torch.randn(nq, nu, device=device))  # D12 = Lambda^{-1} DD12
        self.Y1 = Parameter(torch.zeros(nx, nx, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))
        BIAS = bias
        if (BIAS):
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)

        self.X = Parameter(torch.eye(2*nx+nq, 2*nx+nq, device=device))

        # Initialization of the last Parameters which are constrained:
        self.F = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.D12 = torch.zeros(nq, nu, device=device)
        self.D22 = torch.zeros(ny, nu, device=device)
        self.E = torch.zeros(nx, nx, device=device)
        self.Lambda = torch.zeros(nq, nq, device=device)
        # Choosing the activation function:
        if (sigma == "tanh"):
            self.act = nn.Tanh()
        elif (sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif (sigma == "relu"):
            self.act = nn.ReLU()
        elif (sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def _frame(self):
        pass  # No need for updating weights

    def _check(self) -> bool:
        return True

    def forward(self, x, u):

        # Then solve w_k
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.E)
        dx = x @ (E_inv @ self.F).T + w @ (E_inv @ self.B1).T + u @ (E_inv @ self.B2).T
        if self.feedthrough:
            y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        else:
            y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x, u):
        nb = x.shape[0]  # batch size
        w = torch.zeros(nb, self.nq)
        v = torch.zeros(nb, self.nq)

        # Lambda v_k = C1 x_k + D11 w_k + D12 u_k

        for k in range(0, self.nq):
            v[:, k] = (1/self.Lambda[k, k]) * (x @ self.C1[k, :].T + w @ self.D11[k, :].T 
                                               + u @ self.D12[k, :].T + self.bv[k])
            w[:, k] = self.act(v[:, k])
        return w


class NODE_REN(nn.Module):
    def __init__(self, nx=5, ny=5, nu=5, nq=5,
                 sigma="tanh", epsilon=1.0e-2, mode="c", gamma=1.,
                 device="cpu", bias=False, ni=1., rho=1., alpha=0.0,
                 linear_output=False, feedthrough=True, str_save = None):
        """Base class for Neural Ordinary Differential Equation Recurrent Equilbrium Networks (NODE_RENs).

        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function.
            It is possible to choose: 'tanh','sigmoid','relu','identity'.
            Defaults to "tanh".
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            Defaults to 1.0e-2.
            -mode (str, optional): Property to ensure.
            Possible options:
                -'c'= contractive model
                -'rl2'=L2 lipschitz-bounded,
                -'input_p'=input passive model,
                -'output_p'=output_passive model.
            -gamma (float, optional): If the model is L2 lipschitz bounded (i.e., mode == 'rl2'),
             gamma is the L2 Lipschitz constant. Defaults to 1.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -ni  (float, optional): If the model is input passive (i.e., mode == 'input_p') ,
            ni is the weight coefficient that characterizes the (input passive) supply rate function.
            -rho (float, optional): If the model is output passive (i.e., mode == 'output_p'),
            rho is the weight coefficient that characterizes the (output passive) supply rate function.
            -alpha (float, optional): Lower bound of the Contraction rate.
            If alpha is set to 0, the system continues to be contractive,
            but with a generic (small) rate. Defaults to 0.
            -linear_output (bool, optional): choose if the output is linear,
            i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        self.mode = mode.lower()

        if str_save is None:
            self.str_savepath = os.path.join(os.getcwd(), 'model' + self.mode + '.pkl')
        else:
            self.str_savepath = str_save

        if (self.mode == "general"):
            self.sys = _System_general(nx, ny, nu, nq,
                                       sigma, epsilon, device=device, bias=bias,
                                       linear_output=linear_output)
        else:  # QSR
            if self.mode == "rl2":
                Q = -(1./gamma)*torch.eye(ny, device=device)
                R = (gamma)*torch.eye(nu, device=device)
                S = torch.zeros(nu, ny, device=device)

            elif (self.mode == "input_p"):
                assert nu == ny, "Input and Output dimensions must be equal"
                Q = torch.zeros(ny, device=device)
                R = -2*ni * torch.eye(nu, device=device)
                S = torch.eye(nu, ny, device=device)
            elif (self.mode == "output_p"):
                assert nu == ny, "Input and Output dimensions must be equal"
                Q = -2*rho * torch.eye(nu, device=device)
                R = torch.zeros(nu, device=device)
                S = torch.eye(nu, ny, device=device)
            else:
                raise NameError("The inserted mode is not valid. Please write 'c', 'rl2', 'input_p' or 'output_p'. :(")
            self.nx = nx
            self.nu = nu
            self.ny = ny
            self.nq = nq

            self.sys = _ImplicitQSRNetwork(nx, ny, nu, nq, sigma, epsilon,
                                           S=S, Q=Q, R=R, gamma=gamma, device=device,
                                           bias=bias, alpha=alpha, feedthrough=feedthrough)

    def _frame(self):
        self.sys._frame()  # type: ignore

    def forward(self, u, x):
        dx, y = self.sys(x, u)
        return dx, y

    def check(self):
        self.sys._check()  # type: ignore

    def save(self) -> None:
        torch.save(self.state_dict(), self.str_savepath[:-4] + '.ckpt')

    def load(self, path):
        self.load_state_dict(torch.load(path))
