from typing import Tuple, Optional, Callable
from torch import Tensor
import torch
from cvxpy import Variable, bmat, Minimize, Problem, diag
from cvxpy.error import SolverError
import numpy as np
from .base import LMI



class Lipschitz(LMI):
    r"""
        This class computes an upper bound on the Lipschitz constant for a neural network that can be put into a standard form like

        ..math:
            z = Az + Bx
            w = \Sigma(z)
            y = Cw

        with :math: `\Sigma` being a diagonal operator collecting all activation functions.
        It is assumed the activation functions in :math:`Sigma` are slope restricted in :math:`[0, \beta]`.

        Note : this an algebraic equation, so for this form to be well-posed,
          a sufficient condition can be that A strictly lower triangular but other conditions exist.
          See for example https://arxiv.org/abs/2104.05942

        Attributes:
            A (Tensor) : the interconnexion matrix of size (nz x nz)
            B (Tensor) : the input matrix of size (nz x n_in)
            C (Tensor) : the output matrix of size (n_out x nz)
            extract_matrices (Callable) : a method provided by the model to extract the submatrices for LMI
            beta (float) : maximum admitted slope for activation functions 
                Default = 1
            lip (float) : lipschitz constant for the network
            Lambda_vec (Tensor) : 1-D vector representing the diagonal matrix certificate for LMI feasibility
    """

    ExtractMatricesFn = Callable[[], Tuple[Tensor, ...]]
    def __init__(self,
                 module: torch.nn.Module,
                 extract_matrices: ExtractMatricesFn,
                 beta: float = 1.0,
                 epsilon: float = 1e-6,
                 solver: str = 'MOSEK'):
        
        super(Lipschitz, self).__init__(module, extract_matrices)

        self.beta = beta
        self.epsilon = epsilon
        self.solver = solver

        self.Lambda_vec = None
        self.lip = None
        self.n_in = None
        self.n_out = None
        self.nz = None


    def _update_matrices(self, *args) -> None:

        A, B, C = self.extract_matrices()
        self.A = A
        self.B = B
        self.C = C

        if self.Lambda_vec is None or self.lip is None:
            self.init_(self.A, self.B, self.C, self.beta, self.solver)

    def init_(self,
              A: Optional[Tensor] = None,
              B: Optional[Tensor] = None,
              C: Optional[Tensor] = None,
              beta: float = 1.0,
              solver: str = 'MOSEK'):
        
        if (A is None and B is None and C is None):
            A, B, C = self.extract_matrices()
            
        assert A is not None and B is not None and C is not None
        self.beta = beta
        _, lip, Lambda = Lipschitz.solve(A.detach(), B.detach(), C.detach(), self.beta, solver)

        self.beta = beta
        self.lip = lip
        self.Lambda_vec = torch.diag(Lambda)

        self.n_in = B.shape[1]
        self.n_out = C.shape[0]
        self.nz = A.shape[0]

    def forward(self) -> Tuple[Tensor, ...]:

        assert self.Lambda_vec is not None
        assert self.n_in is not None
        assert self.n_out is not None
        assert self.lip is not None
        # Making Lambda diagonal
        Lambda = torch.diag(self.Lambda_vec)

        M11 = self.lip**2 * torch.eye(self.n_in)
        M12 = - self.B.T @ Lambda
        M13 = torch.zeros((self.n_in, self.n_out))
        M22 = 2 * self.beta * Lambda - Lambda @ self.A - self.A.T @ Lambda
        M23 = - self.C.T
        M33 = torch.eye(self.n_out)

        M1 = torch.cat([M11, M12, M13], dim = 1)
        M2 = torch.cat([M12.T, M22, M23], dim = 1)
        M3 = torch.cat([M13.T, M23.T, M33], dim = 1)

        M = torch.cat([M1, M2, M3], dim = 0)
        return M + self.epsilon*torch.eye(M.shape[0]), Lambda


    @classmethod
    def solve(cls, A: Tensor, B: Tensor, C: Tensor,
              beta: float = 1, solver: str = "MOSEK", tol: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        This class computes an upper bound on the Lipschitz constant for a neural network that can be put into the standard form:

        .. math::

            z = Az + Bx \\
            w = \Sigma(z) \\
            y = Cw

        where :math:`\Sigma` is a diagonal operator collecting all activation functions.

        It is assumed that the activation functions in :math:`\Sigma` are slope-restricted in :math:`[0, \beta]`.

        **Note:** This is an algebraic equation. For the formulation to be well-posed, a sufficient condition is that :math:`A` is strictly lower triangular, although other conditions may apply.

        For more details, see:

        - https://arxiv.org/abs/2104.05942  
        - https://arxiv.org/abs/2005.02929

        **TODO:** For larger networks, solving the LMI can become difficult — this is particularly true for the LBDN architecture. This requires further investigation.
        """


        # Define shapes
        C = C.numpy() # type: ignore
        B = B.numpy() # type: ignore
        A = A.numpy() # type: ignore

        n_in = B.shape[1]
        n_out = C.shape[0]
        nz = A.shape[0]

        lip = Variable()
        Lambda = diag(Variable(nz))
        M11 = lip * np.eye(n_in)
        M12 = - B.T @ Lambda
        M13 = np.zeros((n_in, n_out))
        M22 = 2 * beta * Lambda - Lambda @ A - A.T @ Lambda
        M23 = - C.T
        M33 = np.eye(n_out)

        M = bmat(
            [[M11, M12, M13],
             [M12.T, M22, M23],
             [M13.T, M23.T, M33]]
        )
        nM = M.shape[0]
        constraints = [M - np.eye(nM) * tol >> 0, Lambda - (tol) * np.eye(nz) >> 0, lip - tol >= 0]
        objective = Minimize(lip)

        prob = Problem(objective, constraints=constraints)
        try:
            prob.solve(solver)
        except SolverError:
            prob.solve()  # If MOSEK is not installed then try SCS by default

        if prob.status not in ["infeasible", "unbounded"]:
            assert lip.value is not None
            print(" Lipschitz Constant upper bound (All layer versions): \n")
            print(np.sqrt(lip.value))
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        lip = torch.Tensor(np.array(np.sqrt(lip.value))
                           ).to(dtype=torch.get_default_dtype())
        Lambda = torch.Tensor(Lambda.value).to(dtype=torch.get_default_dtype())
        M = torch.Tensor(M.value).to(dtype=torch.get_default_dtype())
        return M, lip, Lambda
