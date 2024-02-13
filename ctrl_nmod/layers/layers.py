import torch.nn as nn
import torch.nn.functional as F
import geotorch as geo
import torch
from torch.linalg import cholesky, inv
from torch import Tensor


class CustomSoftplus(nn.Softplus):
    def __init__(
        self, beta: int = 1, threshold: int = 20, margin: float = 1e-2
    ) -> None:
        super(CustomSoftplus, self).__init__(beta, threshold)
        self.margin = margin

    def forward(self, x):
        return F.softplus(x, self.beta, self.threshold) + self.margin


def softplus_epsilon(x, epsilon=1e-6):
    return F.softplus(x) + epsilon


class ScaledSoftmax(nn.Softmax):
    def __init__(self, scale=1.0) -> None:
        super(ScaledSoftmax, self).__init__()
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * super().forward(input)


class BetaLayer(nn.Module):
    def __init__(
        self, n_inputs, n_states, n_hidden, actF=nn.Tanh(), func="softplus", tol=0.01
    ) -> None:

        """
        This function initiates a "beta layer" whiches produce a matrix valued function
        beta(x) where beta(x) is invertible and of size n_inputsx n_inputs :
            * beta(x) = U sigma(x) V
            where U,V are orthogonal matrices and
            sigma(x) a one-layer feedforward neural network  :
                pos_func(W_out \sigma(W_in x + b_in)+b_out)
            It is initialized to be Identity matrix
        """
        super(BetaLayer, self).__init__()

        self.nu = n_inputs
        self.nx = n_states
        self.nh = n_hidden
        self.actF = actF
        self.U = nn.Linear(self.nu, self.nx, bias=False)
        nn.init.eye_(self.U.weight)
        geo.orthogonal(self.U, "weight")

        self.V = nn.Linear(self.nx, self.nu, bias=False)
        nn.init.eye_(self.V.weight)
        geo.orthogonal(self.V, "weight")

        self.W_beta_in = nn.Linear(self.nx, self.nh)
        self.W_beta_out = nn.Linear(self.nh, self.nu)
        nn.init.zeros_(self.W_beta_out.weight)
        nn.init.zeros_(self.W_beta_out.bias)

        if func == "softplus":
            self.pos_func = CustomSoftplus(beta=1, threshold=20, margin=tol)
        elif func == "relu":
            self.pos_func = nn.ReLU()
        else:
            raise NotImplementedError("Not implemented yet")

    def forward(self, x):
        sig = self.W_beta_in(x)
        sig = self.actF(sig)
        sig = self.W_beta_out(sig)
        sig = self.pos_func(x)
        return self.U(sig.unsqueeze(-2) @ torch.transpose(self.V.weight, -2, -1))


class InertiaMatrix(nn.Module):
    """
    Implement a SDP matrix function corresponding to an inertia matrix
    depending on a variable q
    """

    def __init__(self, nq, nh, actF=nn.Tanh(), func="softplus", tol=0.01) -> None:

        """
        This function initiates a "beta layer" whiches produce a matrix valued function
        beta(x) where beta(x) is invertible and of size n_inputsx n_inputs :
            * beta(x) = U sigma(x) V
            where U,V are orthogonal matrices and
            sigma(x) a one-layer feedforward neural network  :
            pos_func(W_out \sigma(W_in x + b_in)+b_out)

            It is initialized to be Identity matrix
        """
        super(InertiaMatrix, self).__init__()

        self.nu = nq
        self.nh = nh
        self.actF = actF
        self.U = nn.Linear(self.nu, self.nu, bias=False)
        nn.init.eye_(self.U.weight)
        geo.orthogonal(self.U, "weight")

        self.W_beta_in = nn.Linear(self.nu, self.nh)
        self.W_beta_out = nn.Linear(self.nh, self.nu)
        nn.init.zeros_(self.W_beta_out.weight)
        nn.init.zeros_(self.W_beta_out.bias)

        if func == "softplus":
            self.pos_func = CustomSoftplus(beta=1, threshold=20, margin=tol)
        elif func == "relu":
            self.pos_func = nn.ReLU()
        else:
            raise NotImplementedError("Not implemented yet")

    def forward(self, x):
        sig = self.W_beta_in(x)
        sig = self.actF(sig)
        sig = self.W_beta_out(sig)
        sig = self.pos_func(x)
        return self.U(sig.unsqueeze(-2) @ self.U.weight.transpose(-2, -1))


class CoriolisMatrix(nn.Module):
    def __init__(self, nq, nh, actF=nn.Tanh(), func="softplus", tol=0.01) -> None:

        """
        This function initiates a "beta layer" whiches produce a matrix valued function
        beta(x) where beta(x) is invertible and of size n_inputsx n_inputs :
            * beta(x) = U sigma(x) V
            where U,V are orthogonal matrices and
            sigma(x) a one-layer feedforward neural network  :
            pos_func(W_out \sigma(W_in x + b_in)+b_out)

            It is initialized to be Identity matrix
        """
        super(CoriolisMatrix, self).__init__()

        self.nu = nq
        self.nh = nh
        self.actF = actF
        self.U = nn.Linear(self.nu, self.nu, bias=False)
        nn.init.eye_(self.U.weight)
        geo.orthogonal(self.U, "weight")

        self.W_beta_in = nn.Linear(self.nu, self.nh)
        self.W_beta_out = nn.Linear(self.nh, self.nu)
        nn.init.zeros_(self.W_beta_out.weight)
        nn.init.zeros_(self.W_beta_out.bias)

        if func == "softplus":
            self.pos_func = CustomSoftplus(beta=1, threshold=20, margin=tol)
        elif func == "relu":
            self.pos_func = nn.ReLU()
        else:
            raise NotImplementedError("Not implemented yet")

    def forward(self, x):
        sig = self.W_beta_in(x)
        sig = self.actF(sig)
        sig = self.W_beta_out(sig)
        sig = self.pos_func(x)
        return self.U(sig.unsqueeze(-2) @ torch.transpose(self.U.weight, -2, -1))


class DDLayer(nn.Module):
    def __init__(self, Ui: torch.Tensor) -> None:
        super(DDLayer, self).__init__()
        """
            Ui : inverse of the upper Cholesky decomposition associated to the DD problem
        """
        self.Ui = Ui
        self.act = nn.ReLU()

    def forward(self, M):
        """
        M : linear matrix inequality in Sn+
        """

        Q = self.Ui.T @ M @ self.Ui
        dQ = torch.diag(Q)
        delta_Q = self.act(torch.sum(torch.abs(Q), dim=1) - dQ - torch.abs(dQ))

        DQ = torch.diag(delta_Q)  # (Une) distance à l'ensemble DD+

        return DQ

    def updateU_(self, M):
        """
        From the current M value we update U to update search region in DD+
        If correct : DDLayer(updateU_(M)) = I
        """
        # Q = self(M)
        # M_next = inv(self.Ui.T) @ Q @ inv(self.Ui)
        # assert torch.all(M_next == M)
        self.Ui = inv(cholesky(M).mH)
