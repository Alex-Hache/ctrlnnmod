from typing import Tuple, List, Optional
from torch import Tensor
import torch
import torch.nn as nn
from cvxpy import Variable, bmat, hstack, vstack, Minimize, Problem
from ..linalg.utils import block_diag
import numpy as np
from base import LMI


class LipschitzLMI(LMI):
    def __init__(self, weights: List[Tensor], beta: float = 1.0,
                 T: Optional[Tensor] = None, lip: Optional[float] = None, epsilon: float = 1e-6):
        super().__init__()
        self.weights = weights
        self.beta = beta
        self.epsilon = epsilon

        if T is None:
            T, eta, _ = LipschitzLMI.solve(weights=weights, beta=beta)

        if lip is not None:
            self.lip = nn.Parameter(torch.Tensor(lip))
            if self.lip < eta:
                self.lip = eta
        else:
            self.T = nn.Parameter(T)
            self.lip = nn.Parameter(eta)

    def forward(self) -> Tensor:
        weights = self.weights
        beta = self.beta

        n_in = weights[0].shape[1]
        n_hidden = [w.shape[0] for w in weights[:-1]]
        n_out = weights[-1].shape[0]

        # Since self.T is already block diagonal
        Ft_top = torch.cat((torch.zeros_like(self.T), beta * self.T), dim=1)
        Ft_bottom = torch.cat((beta * self.T, -2 * self.T), dim=1)
        Ft = torch.cat((Ft_top, Ft_bottom), dim=0)

        Ws = [w for w in weights[:-1]]
        W_block = torch.block_diag(*Ws)
        A = torch.cat((W_block, torch.zeros(W_block.shape[0], n_hidden[-1])), dim=1)

        I_B = torch.eye(sum(n_hidden))
        B = torch.cat((torch.zeros(I_B.shape[0], n_in), I_B), dim=1)
        AB = torch.cat((A, B), dim=0)
        LMI = AB.t() @ Ft @ AB

        LMI_schur = torch.block_diag(LMI, torch.zeros(n_out, n_out))

        L = -self.lip * torch.eye(n_in)

        b11 = torch.zeros(n_hidden[-1], n_hidden[-1])
        b12 = weights[-1].t()
        b21 = weights[-1]
        b22 = -torch.eye(n_out)
        bf_top = torch.cat((b11, b12), dim=1)
        bf_bottom = torch.cat((b21, b22), dim=1)
        bf = torch.cat((bf_top, bf_bottom), dim=0)

        if sum(n_hidden[:-1]) > 0:
            inter = torch.zeros(sum(n_hidden[:-1]), sum(n_hidden[:-1]))
            part2 = torch.block_diag(L, inter, bf)
        else:
            part2 = torch.block_diag(L, bf)

        M = LMI_schur + part2

        return M

    @classmethod
    def solve(cls, weights: List[Tensor], beta: float = 1, solver: str = "MOSEK", tol: float = 1e-6) -> Tuple[Tensor, Tensor, Tensor]:
        r'''
            This function solves the Linear Matrix Inequality for
            estimating an upper bound on the Lipschitz constant of
            a feedforward neural network without skip connections.
            https://arxiv.org/abs/2005.02929
            params :
                * weights : a list of neural network weights
                * beta : the maximum slope of activation functions
        '''
        weights = [w.cpu()
                   for w in weights]  # Ensure weights are on CPU for cvxpy

        n_in = weights[0].shape[1]
        n_hidden = [w.shape[0] for w in weights[:-1]]
        n_out = weights[-1].shape[0]

        Ts = [Variable((n_h, n_h), diag=True) for n_h in n_hidden]
        T = block_diag(Ts)
        Ft = bmat([[np.zeros(T.shape), beta * T], [beta * T, -2 * T]])
        Ws = [weight.detach().numpy() for weight in weights[:-1]]
        W = block_diag(Ws)
        A = hstack([W, np.zeros((W.shape[0], n_hidden[-1]))])

        I_B = np.eye(sum(n_hidden))
        B = hstack([np.zeros((I_B.shape[0], n_in)), I_B])
        AB = vstack([A, B])
        LMI = AB.T @ Ft @ AB

        LMI_schur = block_diag([LMI, np.zeros((n_out, n_out))])

        lip = Variable()
        L = -lip * np.eye(n_in)

        b11 = np.zeros((n_hidden[-1], n_hidden[-1]))
        b12 = weights[-1].T.detach().numpy()
        b21 = weights[-1].detach().numpy()
        b22 = -np.eye(n_out)
        bf = bmat([[b11, b12], [b21, b22]])

        dim_inter = sum(n_hidden[:-1])
        if dim_inter > 0:
            inter = np.zeros((dim_inter, dim_inter))
            part2 = block_diag([L, inter, bf])
        else:
            part2 = block_diag([L, bf])

        M = LMI_schur + part2

        nM = M.shape[0]
        nT = T.shape[0]
        constraints = [M << -np.eye(nM) * tol, T - (tol) * np.eye(nT) >> 0, lip - tol >= 0]
        objective = Minimize(lip)

        prob = Problem(objective, constraints=constraints)
        prob.solve(solver)

        if prob.status not in ["infeasible", "unbounded"]:
            print(" Lipschitz Constant upper bound (All layer versions): \n")
            print(np.sqrt(lip.value))
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        lip = torch.Tensor(np.array(np.sqrt(lip.value))
                           ).to(dtype=torch.float32)
        T = torch.Tensor(T.value).to(dtype=torch.float32)
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        return T, lip, M
