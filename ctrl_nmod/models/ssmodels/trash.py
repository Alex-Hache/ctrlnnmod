
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
        self.epsilon = 0.0
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
        self.Y1 = Parameter(torch.zeros(nx, nx, device=device)*std)
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

        self.X = Parameter(torch.eye(2*nx+nq, 2*nx+nq, device=device))

        # Initialization of the last Parameters:
        self.F = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.D12 = torch.zeros(nq, nu, device=device)
        self.D22 = torch.zeros(ny, nu, device=device)
        self.E = torch.zeros(nx, nx, device=device)
        self.Lambda = torch.zeros(nq, nq, device=device)

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
        # P = 0.5*F.linear(self.Pstar, self.Pstar)+self.epsilon*torch.eye(self.nx, device=self.device)
        M = F.linear(self.X3, self.X3) + (self.Y3 - self.Y3.T +
                                          (self.epsilon * torch.eye(self.s, device=self.device)))
        if self.feedthrough:  # Only update D22 if direct feedthrough
            M_tilde = F.linear(torch.eye(self.s, device=self.device) - M,
                               torch.inverse(torch.eye(self.s, device=self.device)+M).T)
            M_tilde = M_tilde[0:self.ny, 0:self.nu]

            self.D22 = self.gamma*M_tilde

        R_capital = self.R - (1/self.gamma)*F.linear(self.D22.T, self.D22.T)
        C2_tilde = (self.D22.T @ self.Q) @ self.C2
        D21_tilde = (self.D22.T @ self.Q) @ self.D21 - self.DD12.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)

        vec_C2_D21 = torch.cat([self.C2.T, self.D21.T, torch.zeros((self.nx, self.ny))], 0)
        Hs1 = F.linear(self.X, self.X)  # X X^T
        Hs2 = self.epsilon*torch.eye(2*self.nx+self.nq, device=self.device)
        H_cvx = Hs1 + Hs2

        Hs3 = F.linear(V_tilde, torch.inverse(R_capital).T)
        Hs4 = F.linear(Hs3, V_tilde)
        diss = Hs4 + np.sqrt(1/self.gamma)*F.linear(vec_C2_D21, vec_C2_D21)

        H = H_cvx + diss

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
        self.D12 = lambda_inv @ self.DD12

    def forward(self, xi, u):
        Nb = xi.shape[0]
        # By = F.linear(torch.ones(Nb, 1, device=self.device), self.by)

        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(Nb, self.nq, device=self.device)
        v = (1/self.Lambda[0:1, 0:1]) * (xi @ self.C1[0:1, :].T +
                                      torch.ones((Nb, 1), device=self.device) *self.bv[0:1] +
                                     u @ self.D12[0:1, :].T)

        v = ((1/self.Lambda[0, 0]) * (xi @ self.C1[0, :].T + u @ self.D12[0, :].T + self.bv[0])).unsqueeze(1)
        w = w + self.act(v) @ vec.T
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = ((1/self.Lambda[i, i]) * ((xi @ self.C1[i, :].T) + (w @ self.D11[i, :].T) +
                 self.bv[i]*torch.ones(Nb, device=self.device) +
                 u @ self.D12[i, :].T)).unsqueeze(1)  # + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v), vec)
        xi_ = torch.inverse(self.E) @ (F.linear(xi, self.F) + F.linear(w, self.B1) + F.linear(torch.ones(Nb, 1), self.bx) + F.linear(u, self.B2))
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_  # ,yi

    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        Nb = xi.shape[0]
        By = F.linear(torch.ones(Nb, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(Nb, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) +
             self.bv[0]*torch.ones(Nb, device=self.device) + F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, self.D11[i, :]) +
                 self.bv[i]*torch.ones(Nb, device=self.device) + F.linear(u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By

    def check_H_cvx(self):  # H_cvx = X X.T + eps*I
        H11 = self.E + self.E.T - (1/self.alpha**2) * self.P

        H21 = -self.C1
        H31 = self.F
        H22 = 2*self.Lambda - self.Lambda @ self.D11 - self.D11.T @ self.Lambda
        H32 = self.B1
        H33 = self.P

        H1 = torch.cat([H11, H21.T, H31.T], dim=1)
        H2 = torch.cat([H21, H22, H32.T], dim=1)
        H3 = torch.cat([H31, H32, H33], dim=1)

        H = torch.cat([H1, H2, H3], dim=0)
        return H

    def check(self):
        
            On construit 20 à partir des poids du réseau
            on vérifie que 20 est bien égale à XX^T + 28b
        H11 = self.E + self.E.T - (1/self.alpha**2) * self.P

        H21 = -self.C1
        H12 = H21.T

        H31 = self.S @ self.C2
        H13 = H31.T

        H22 = 2*self.Lambda - self.Lambda @  self.D11 - self.D11.T @ self.Lambda

        H32 = self.S @ self.D21 - self.DD12.T
        H23 = H32.T
        H33 = self.R + self.S @ self.D22 + (self.S @ self.D22).T

        H1 = torch.cat([H11, H12, H13], dim=1)
        H2 = torch.cat([H21, H22, H23], dim=1)
        H3 = torch.cat([H31, H32, H33], dim=1)

        H = torch.cat([H1, H2, H3], dim=0)

        # QSR dissipativity part
        J = torch.cat([self.F.T, self.B1.T, self.B2.T], dim=0)
        K = torch.cat([self.C2.T, self.D21.T, self.D22.T], dim=0)
        diss = -J @ torch.inverse(self.P) @ J.T + K @ self.Q @ K.T

        M = H + diss
        return H, diss, M, isSDP(M)