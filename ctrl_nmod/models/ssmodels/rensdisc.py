import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter


class _System_robust_L2_bound(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, S, Q, R,
                 gamma, device, bias=False, alpha=0.0, feedthrough=True):
        """Used by the upper class NODE_REN to guarantee the model to be L2 Lipschitz
        bounded in its input-output mapping (and thus, robust). It should not be used by itself.

        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu',
                    'identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices
                     to be positive definitive.
            -S (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x ny
            -Q (torch.tensor): Weight matrix used in the supply rate. Dimensions: ny x ny
            -R (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x nu
            -gamma (float): L2 Lipschitz constant.
            -device (string): device to be used for the computations using Pytorch
                    (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0,
             the system continues to be contractive, but with a generic (small) rate. Defaults to 0.
            """
        super().__init__()
        # Dimensions of Inputs, Outputs, States

        self.nx = nx        # no. internal-states
        self.ny = ny        # no. output
        self.nu = nu        # no. inputs
        self.nq = nq        # no. non-linear states
        self.s = np.max((nu, ny))
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.alpha = alpha
        self.feedthrough = feedthrough
        std = 0.02

        # Initialization of the Free Matrices:
        self.Pstar = Parameter(torch.randn(nx, nx, device=device)*std)
        self.Chi = Parameter(torch.randn(nx, nq, device=device)*std)

        # Initialization of the Weights:
        self.DD12 = Parameter(torch.randn(nq, nu, device=device)*std)
        self.X3 = Parameter(torch.randn(self.s, self.s, device=device)*std)
        self.Y3 = Parameter(torch.randn(self.s, self.s, device=device)*std)
        self.Y1 = Parameter(torch.randn(nx, nx, device=device)*std)
        self.B2 = Parameter(torch.randn(nx, nu, device=device)*std)
        self.C2 = Parameter(torch.randn(ny, nx, device=device)*std)
        self.D21 = Parameter(torch.randn(ny, nq, device=device)*std)
        BIAS = bias
        if (BIAS):
            self.bx = Parameter(torch.randn(nx, 1, device=device)*std)
            self.bv = Parameter(torch.randn(nq, 1, device=device)*std)
            self.by = Parameter(torch.randn(ny, 1, device=device)*std)
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)

        self.X = Parameter(torch.randn(nx+nq, nx+nq, device=device)*std)

        # Initialization of the last Parameters:
        self.A = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.D12 = torch.zeros(nq, nu, device=device)
        self.D22 = torch.zeros(ny, nu, device=device)

        self.Lq = np.sqrt(1/gamma)*torch.eye(ny, device=device)
        self.Lr = np.sqrt(gamma)*torch.eye(nu, device=device)
        self.R = R
        self.Q = Q
        self.S = S

        self.updateParameters()             # Update of: A, B1, C1, D11
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

    def updateParameters(self):
        P = 0.5*F.linear(self.Pstar, self.Pstar)+self.epsilon*torch.eye(self.nx, device=self.device)
        M = F.linear(self.X3, self.X3) + (self.Y3 - self.Y3.T +
                                          (self.epsilon * torch.eye(self.s, device=self.device)))
        if self.feedthrough:  # Only update D22 if direct feedthrough
            M_tilde = F.linear(torch.eye(self.s, device=self.device) - M,
                               torch.inverse(torch.eye(self.s, device=self.device)+M).T)
            M_tilde = M_tilde[0:self.ny, 0:self.nu]

            self.D22 = self.gamma*M_tilde

        R_capital = self.R - (1/self.gamma)*F.linear(self.D22.T, self.D22.T)
        C2_tilde = (self.D_22.T @ self.Q) @ self.C2
        D21_tilde = (self.D22.T @ self.Q) @ self.D21 - self.DD12.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)

        vec_C2_D21 = torch.cat([self.C2.T, self.D21.T], 0)
        H = ((F.linear(self.X, self.X) +
              self.epsilon*torch.eye(self.nx+self.nq, device=self.device) +
              F.linear(F.linear(V_tilde, torch.inverse(R_capital).T), V_tilde)) +
             np.sqrt(1/self.gamma)*F.linear(vec_C2_D21, vec_C2_D21))

        # Partition of H in --> [H1 H2;H3 H4]
        h1, h2 = torch.split(H, [self.nx, self.nq], dim=0)  # you split the matrices in two big rows
        H1, H2 = torch.split(h1, [self.nx, self.nq], dim=1)  # you split each big row in two chunks
        H3, H4 = torch.split(h2, [self.nx, self.nq], dim=1)

        Y = -0.5*(H1 + self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P), Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        self.C1 = F.linear(torch.inverse(Lambda), self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P), Z.T)
        self.D12 = F.linear(torch.inverse(Lambda), self.DD12.T)

        # # Check the LMI for robustness is verified.
        # pr11 = -Y -Y.T
        # pr12 = -Z - self.Chi
        # pr13 = -F.linear(P,self.B2.T)
        # pr21= pr12.T
        # pr22=H4
        # pr23=-F.linear(Lambda,self.D12.T)
        # pr31=pr13.T
        # pr32=pr23.T
        # pr33=self.R
        # temp = torch.cat([self.C2.T,self.D21.T,self.D22.T],0)
        # d_temp = -np.sqrt(1/self.gamma)*F.linear(temp,temp)
        # LMI = torch.cat([
        #     torch.cat([pr11,pr12,pr13],1),
        #     torch.cat([pr21,pr22,pr23],1),
        #     torch.cat([pr31,pr32,pr33],1)],0)+d_temp
        # results = torch.linalg.eigvals(LMI)
        # print(results)
        # print("")

    def forward(self, xi, u):
        n_initial_states = xi.shape[0]
        # By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) +
             self.bv[0] * torch.ones(n_initial_states, device=self.device) +
             F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, self.D11[i, :]) +
                 self.bv[i]*torch.ones(n_initial_states, device=self.device) +
                 F.linear(u, self.D12[i, :])).unsqueeze(1)  # + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v), vec)
        xi_ = (F.linear(xi, self.A) + F.linear(w, self.B1) +
               F.linear(torch.ones(n_initial_states, 1), self.bx) + F.linear(u, self.B2))
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_  # ,yi

    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) +
             self.bv[0]*torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, self.D11[i, :]) +
                 self.bv[i]*torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By


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
        std = .7         # standard deviation used to draw randomly the initial weights of the model.
        # Initialization of the Free Matrices:
        # self.Pstar = Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Chi = Parameter(torch.randn(nx,nq,device=device)*std)
        # Initialization of the Weights:
        # self.Y1 = Parameter(torch.randn(nx,nx,device=device)*std)
        self.A = Parameter(torch.randn(nx, nx, device=device)*std)
        self.B1 = Parameter(torch.randn(nx, nq, device=device)*std)
        self.B2 = Parameter(torch.randn(nx, nu, device=device)*std)
        self.C1 = Parameter(torch.randn(nq, nx, device=device)*std)
        # self.D11_coefficients = Parameter(torch.randn(nq,device=device)*std)
        self.D11 = Parameter(torch.randn(nq, nq, device=device)*std)
        self.D12 = Parameter(torch.randn(nq, nu, device=device)*std)
        self.C2 = Parameter(torch.randn(ny, nx, device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny, nq, device=device)
        else:
            self.D21 = Parameter(torch.randn(ny, nq, device=device)*std)
        self.D22 = Parameter(torch.randn(ny, nu, device=device)*std)
        BIAS = bias
        if (BIAS):
            self.bx = Parameter(torch.randn(nx, 1, device=device)*std)
            self.bv = Parameter(torch.randn(nq, 1, device=device)*std)
            self.by = Parameter(torch.randn(ny, 1, device=device)*std)
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)
        # self.X = Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        # Initialization of the last Parameters:
        # self.Y= torch.zeros(nx,nx)
        # self.P = torch.zeros(nx,nx,device=device)
        # self.alpha= alpha
        # self.updateParameters()             #Update of: A, B1, C1, D11
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

    def forward(self, t, xi, u):
        n_initial_states = xi.shape[0]
        # By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) +
             F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) +
                 F.linear(w, torch.tril(self.D11, -1)[i, :]) +
                 self.bv[i] * torch.ones(n_initial_states, device=self.device) +
                 F.linear(u, self.D12[i, :])).unsqueeze(1)  # + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v), vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(
            torch.ones(n_initial_states, 1, device=self.device), self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_  # ,yi

    def updateParameters(self):
        # A general NodeREN does not require any additional step.
        pass

    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) +
             F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) +
                 F.linear(w, torch.tril(self.D11, -1)[i, :]) +
                 self.bv[i] * torch.ones(n_initial_states, device=self.device) +
                 F.linear(u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By


class NODE_REN(nn.Module):
    def __init__(self, nx=5, ny=5, nu=5, nq=5,
                 sigma="tanh", epsilon=1.0e-2, mode="c", gamma=1.,
                 device="cpu", bias=False, ni=1., rho=1., alpha=0.0,
                 linear_output=False, feedthrough=True):
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
        self.nfe = 0
        if (self.mode == "general"):
            self.sys = _System_general(nx, ny, nu, nq,
                                       sigma, epsilon, device=device, bias=bias,
                                       linear_output=linear_output)
        elif (self.mode == "rl2"):
            Q = -(1./gamma)*torch.eye(ny, device=device)
            R = (gamma)*torch.eye(nu, device=device)
            S = torch.zeros(nu, ny, device=device)
            self.sys = _System_robust_L2_bound(nx, ny, nu, nq, sigma, epsilon,
                                               S=S, Q=Q, R=R, gamma=gamma, device=device,
                                               bias=bias, alpha=alpha, feedthrough=feedthrough)
        else:
            raise NameError("The inserted mode is not valid. Please write 'c', 'rl2', 'input_p' or 'output_p'. :(")

    def updateParameters(self):
        self.sys.updateParameters()

    def forward(self, u, x):
        self.nfe += 1
        dx = self.sys(x, u)
        return dx

    def output(self, u, x):
        yt = self.sys.output(x, u)
        return yt

    @property
    def nfe(self):
        return self._nfe

    @nfe.setter
    def nfe(self, value):
        self._nfe = value
