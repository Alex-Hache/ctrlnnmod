'''
    This module include some parameterized matrices for constrained network weights
'''
import warnings
from typing import Tuple, Union

import numpy as np
import torch
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from ctrl_nmod.layers.layers import softplus_epsilon, ScaledSoftmax
from torch import Tensor
from torch.linalg import matrix_exp
from torch.nn import Module
from torch.nn.parameter import Parameter

from ctrl_nmod.linalg.utils import cayley, isSDP, is_alpha_stable

# TODO Implement a parent class parameterized matrix with the following methods
#   __init__
#  solve
#  forward
#  eval_

# TODO Make the classes inherit from the Linear module in torch


class SkewSymetric(Module):
    '''
        This function implements a parameterization of skew symmetric weights
        Methods:
            - __init__ : default constructor
            - forward : matrix multiplication
            - _check : check if the weight is skew-symmetric
            - update : project the weight onto the set of skew symmetric matrices
    '''
    def __init__(self, arg: Union[int, Tensor]) -> None:
        '''
            Constructor for SkewSymetric class :
            arg : It accepts either :
                n : an integer specifying the size of the matrix
                S : a tensor that is skew symmetric
                if the provided tensor is not skew symmetric then we take its skew symmetric part
        '''
        super(SkewSymetric, self).__init__()
        if isinstance(arg, int):
            self.n = arg
            X = torch.rand((self.n, self.n))
            self.weight = Parameter(0.5*(X - X.T)).requires_grad_(True)
        elif isinstance(arg, Tensor):
            assert arg.shape[0] == arg.shape[1], "Provided tensor is not square"
            self.n = arg.shape[0]
            if torch.all(arg == -arg.T):
                self.weight = Parameter(arg).requires_grad_(True)
            else:  # Taking skew symmetric par (X -X^T)/2
                warnings.warn("Provided tensor not skew taking its projection")
                self.weight = Parameter(0.5*(arg - arg.T)).requires_grad_(True)
        else:
            raise ValueError("Expected int or tensor")

    def forward(self, x: Tensor) -> Tensor:
        return x@(0.5*(self.weight - self.weight.T).T)  # Assuming batch dimension first

    def eval_(self) -> Tensor:
        return (0.5*(self.weight - self.weight.T))


class Orthogonal(Module):
    '''
        Defines an (semi-)orthogonal matrix in Q \in \mathbb{R}^{m \times p}:
        if p > m Q has orthonormal rows -> Q^TQ = I_m
        if p <= m Q has orthonormal columns -> Q Q^T = I_p
    '''
    def __init__(self, arg: Union[Tuple[int, int], Tensor], param='expm') -> None:
        super(Orthogonal, self).__init__()
        self.param = param
        if isinstance(arg, tuple):
            self.p, self.m = arg[0], arg[1]
            if param == 'expm':
                size = max(self.p, self.m)
                self.skew = SkewSymetric(size)
                # The exponential of a skew symmetric matrix is in SO(n)
                W = matrix_exp(self.skew.eval_())
                self.weight = Parameter(W[:self.m, :self.p]).requires_grad_(True)
            elif param == 'cayley':
                W = torch.rand((self.m, self.p))
                self.weight = Parameter(cayley(W)).requires_grad_(True)
            else:
                raise TypeError("Please specify either 'expm" or 'cayley')
        elif isinstance(arg, Tensor):
            self.p, self.m = arg.shape[0], arg.shape[1]
            self.weight = Parameter(arg).requires_grad_(True)
        else:
            raise TypeError("Expected a shape or a tensor")

    def forward(self, x) -> Tensor:
        if self.param == 'cayley':
            return x@(cayley(self.weight).T)
        elif self.param == 'expm':
            return x@((matrix_exp(self.skew.eval_())[:self.m, :self.p]).T)
        else:
            raise BaseException("Undefined parameterization")

    def eval_(self) -> Tensor:
        if self.param == 'cayley':
            return (cayley(self.weight))
        elif self.param == 'expm':
            return (matrix_exp(self.skew.eval_())[:self.m, :self.p])
        else:
            raise BaseException("Undefined parameterization")


class PositiveDefinite(Module):
    '''
        This function is module that returns a positive definite matrix
        this function accepts either
            * n : an integer defining the size of the matrix
            * A : a matrix itself being positive definite
    '''
    def __init__(self, arg: Union[int, Tensor], param: str = 'svd', epsilon=1e-4) -> None:
        ''''''
        super(PositiveDefinite, self).__init__()
        self.epsilon = epsilon
        self.param = param
        if isinstance(arg, int):
            self.n = arg
            if param == 'svd':  # Parameterized using singular value decomposition
                self.P = Orthogonal((self.n, self.n))
                self.diag = Parameter(torch.rand((self.n))).requires_grad_(True)
                self.weight = (self.P.eval_() @
                               (torch.diag(softplus_epsilon(self.diag, self.epsilon)) @
                               self.P.eval_().T))
            elif param == 'square':  # Parameterized as XX^T
                self.weight = Parameter(torch.rand((self.n, self.n))).requires_grad_(True)
            else:
                raise ValueError("Please choose either 'svd' or 'square'")
        elif isinstance(arg, Tensor):
            self.n = arg.shape[0]
            if not isSDP(arg):
                raise ValueError("Provided Tensor is not positive definite")
            self.weight = Parameter(arg).requires_grad_(True)
        else:
            raise TypeError("Expected an integer or Tensor")

    def forward(self, x) -> Tensor:
        if self.param == 'svd':
            return x@((self.P.eval_() @ torch.diag(softplus_epsilon(self.diag, self.epsilon))
                       @ self.P.eval_().T).T)
        elif self.param == 'square':
            return x@((self.weight @ self.weight.T + self.epsilon * torch.eye(self.n)).T)
        else:
            raise ValueError("Wrong parameterization")

    def eval_(self) -> Tensor:
        if self.param == 'svd':
            return (self.P.eval_() @ torch.diag(softplus_epsilon(self.diag, self.epsilon)) @ self.P.eval_().T)
        elif self.param == 'square':
            return (self.weight @ self.weight.T + self.epsilon * torch.eye(self.n))
        else:
            raise ValueError("Wrong parameterization")


class AlphaStable(Module):
    '''
        This module produces a direct parametrization of the A matrix to be alpha-stable
        i.e. the real part of its eigenvalue is lower than -\alpha < 0
        A = P^{-1}(Q/2 + S) - \alpha I
        with P, Q > 0 and S skew symmetric matrix
        The A matrix is then solution of the Lyapunov equation :
        A^T P + PA + 2 alpha P = -Q
    '''

    def __init__(self, arg: Union[int, Tensor], alpha=1.0) -> None:
        ''''
            Default constructor for alpha-stable matrices
            params :
                * arg : either the size of the a matrix either an alpha-stable A matrix
        '''
        super(AlphaStable, self).__init__()
        self.alpha = alpha
        if isinstance(arg, int):
            self.n = arg
            self.Pinv = PositiveDefinite(self.n)
            self.Q = PositiveDefinite(self.n)
            self.S = SkewSymetric(self.n)
        elif isinstance(arg, Tensor):
            self.n = arg.shape[0]
            if not is_alpha_stable(arg, self.alpha):
                raise ValueError("Provided Tensor is not alpha stable")
            Q, P = self.solve_(arg)
            self.Q = PositiveDefinite(Q)
            self.Pinv = PositiveDefinite(torch.linalg.inv(P))
            self.S = SkewSymetric(0.5*Q + P @ arg)  # S is solution of A^T P + PA
        else:
            raise TypeError("Expected an integer or Tensor")

    def solve_(self, A: Tensor, epsilon=1e-6, solver="MOSEK"):
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
        P_inv = self.Pinv.eval_()
        Q = self.Q.eval_()
        S = self.S.eval_()
        return x@((P_inv@(-0.5 * Q + S)
                   - self.alpha*torch.eye(self.n)).T)

    def eval_(self):
        '''
            Evaluate the value of the corresponding paremeterized matrix A
        '''
        P_inv = self.Pinv.eval_()
        Q = self.Q.eval_()
        S = self.S.eval_()
        return (P_inv@(-0.5 * Q + S)
                - self.alpha*torch.eye(self.n))


class FixedTrace(Module):
    '''
        Fixed trace matrix
    '''
    def __init__(self, arg: Union[int, Tensor], trace=1.0) -> None:
        super(FixedTrace, self).__init__()
        self.trace = trace
        self.scaler = ScaledSoftmax(self.trace)

        if isinstance(arg, int):
            self.n = arg
            self.weight = Parameter(torch.rand((self.n, self.n))).requires_grad_(True)
        elif isinstance(arg, Tensor):
            self.n = arg.shape[0]
            self.weight = Parameter(arg).requires_grad_(True)
        else:
            raise TypeError("Expected an integer or Tensor")

        self.diag = torch.rand((self.n))

    def forward(self, x):
        w_diag = torch.diag(self.weight)
        return x @ ((self.weight - w_diag + self.scaler(self.diag)).T)

    def eval_(self):
        w_diag = torch.diag(self.weight)
        return (self.weight - w_diag + self.scaler(self.diag))
