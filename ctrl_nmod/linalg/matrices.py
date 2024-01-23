'''
    This module include some parameterized matrices for constrained network weights
'''
import geotorch as geo
import numpy as np
import torch
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from torch.nn import Module, init
from torch.nn.parameter import Parameter


class StableA(Module):
    '''
        This module produces a direct parametrization of the A matrix to be alpha-stable
        A = P^{-1}(Q/2 + S) - \alpha I
        with P, Q > 0 and S skew symmetric matrix
    '''

    def __init__(self, nx, alpha=1.0, A: torch.Tensor = torch.empty()) -> None:
        super(StableA, self).__init__()
        self.alpha = alpha
        self.nx = nx
        if A.numel() != 0:  # Not empty tensor
            Q, P = self.solve_lmi(A)
            self.Q = Parameter(Q)
            self.Pinv = Parameter(torch.inverse(P))
            self.S = Parameter(init.normal_(torch.empty(self.nx, self.nx)))
            # register P,Q,S variables to be on specified manifolds
            geo.positive_definite(self, 'Pinv')
            geo.positive_definite(self, 'Q')
            geo.skew(self, 'S')  # type:ignore
        else:
            self.Q = Parameter(init.normal_(torch.empty(self.nx, self.nx)))
            self.Pinv = Parameter(init.normal_(torch.empty(self.nx, self.nx)))
            self.S = Parameter(init.normal_(torch.empty(self.nx, self.nx)))
            # TODO : how to enforce specific values from a first LMI resolution ?

    def solve_lmi(self, A: torch.Tensor, epsilon=1e-6, solver="MOSEK"):
        print("Initializing Lyapunov matrix \n")

        A = A.detach().numpy()
        P = Variable((self.nx, self.nx), 'P', PSD=True)
        nx = self.nx

        M = A.T@P + P@A + 2 * self.alpha * P  # type: ignore
        constraints = [M << -epsilon*np.eye(nx), P - (epsilon)*np.eye(nx) >> 0]  # type: ignore
        objective = Minimize(0)  # Feasibility problem

        prob = Problem(objective, constraints=constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)

        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return torch.Tensor(-M.value), torch.Tensor(P.value)

    def forward(self, x):
        return x@((self.Pinv@(-0.5*self.Q + self.S) - self.alpha*torch.eye(self.nx)).T)

    def eval_(self):
        '''
            Evaluate the value of the corresponding paremeterized matrix A
        '''
        return (self.Pinv@(-0.5*self.Q + self.S) - self.alpha*torch.eye(self.nx))
