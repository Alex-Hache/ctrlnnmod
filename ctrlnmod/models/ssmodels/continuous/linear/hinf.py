import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.module import _addindent
from torch.nn import Module
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from ctrlnmod.linalg.utils import sqrtm

from ctrlnmod.utils import FrameCacheManager
from ctrlnmod.linalg import project_onto_stiefel
from ctrlnmod.lmis.hinf import HInfCont
from .base import SSModel
from ctrlnmod.linalg.utils import solve_riccati_torch, is_positive_definite, schur


class L2BoundedLinear(SSModel):
    r"""
        Create a linear continuous-time state-space model with a prescribed L2 gain and alpha stability.
        math::   
            \dot{x} &= Ax + Bu \\
            y &= Cx

        Attributes:
            nu (int): Number of inputs
            ny (int): Number of outputs
            nx (int): Number of states
            gamma (Tensor): L2 gain
            alpha (float): alpha stability margin
            param (str): Parameterization method ('sqrtm' or 'ricccati')
            epsilon (float): Small positive number for positive definiteness
    """
    def __init__(self, nu: int, ny: int, nx: int, gamma: float, alpha: float = 0.0,
                 param: str = 'sqrtm', epsilon=1e-3) -> None:
        super(L2BoundedLinear, self).__init__(nu, ny, nx)

        assert param in ['sqrtm', 'riccati'], "param must be 'sqrtm' or 'riccati'"
        assert alpha >= 0.0, "alpha must be non-negative"
        if param == 'riccati':
            assert alpha == 0.0, "alpha must be 0.0 for Riccati parameterization"


        self.gamma = gamma
        self.Ix = torch.eye(nx)
        self.eps = epsilon
        self.param = param
        self.alpha = alpha

        self.P = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.G = Parameter(nn.init.xavier_normal_(torch.empty(self.ny, self.nx)))
        

        # Register relevant manifolds
        geo.positive_definite(self, 'P')
        geo.skew_symmetric(self, 'S')

        if param == 'sqrtm':
            self.Q = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
            geo.positive_definite(self, 'Q')
            self.H = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nu)))
            geo.orthogonal(self, 'H')
            # M matrix will store the result of the LMI
            self.M = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx))) 
            geo.positive_definite(self, 'M')
        else:
            self.B = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nu)))

        self._frame_cache = FrameCacheManager()
    
    def __repr__(self):
        super_repr = super(L2BoundedLinear, self).__repr__() + "\n"
        super_repr += f"gamma={self.gamma}, alpha={self.alpha}, param={self.param}"
        return super_repr
    
    def __str__(self):
        return self.__repr__()
    

    def forward(self, u, x, d=None) -> tuple[Tensor, Tensor]:
        """
            Forward pass of the linear system.
            Args:
                u (Tensor): Input tensor of shape (batch_size, nu)
                x (Tensor): State tensor of shape (batch_size, nx)
                d (Tensor, optional): Exogenous input tensor of shape (batch_size, nd). Default is None.
            Returns:
                dx (Tensor): State derivative tensor of shape (batch_size, nx)
                y (Tensor): Output tensor of shape (batch_size, ny)
        """
        A, B, C = self._frame()
        dx = x @ A.T + u @ B.T

        y = x @ C.T
        return dx, y

    def _frame(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.param == 'sqrtm':
            A, B, C = self._frame_sqrt()
        else:
            A, B, C = self._frame_riccati()
        return A, B, C
    


    def _frame_sqrt(self) -> tuple[Tensor, ...]:
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        A = (-0.5 * (self.Q + self.G.T @ self.G + self.eps * self.Ix) + self.S) @ self.P -0.5* torch.inverse(self.P) @ self.M - self.alpha * self.Ix
        B = (self.gamma) * sqrtm(self.Q) @ self.H # type: ignore
        C = self.G @ self.P
        if self._frame_cache.is_caching:    
            self._frame_cache.cache = (A, B, C)
        return A, B, C
    
    def _frame_riccati(self) -> tuple[Tensor, ...]:
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        B = self.B

        A = (-0.5 * (1/self.gamma**2 * (B @ B.T) + self.G.T @ self.G) + self.S) @ self.P - self.alpha * self.Ix
        C = self.G @ self.P
        if self._frame_cache.is_caching:    
            self._frame_cache.cache = (A, B, C)
        return A, B, C


    def _right_inverse(self, A0: Tensor, B0: Tensor, C0: Tensor, gamma: float, alpha=0.0):
        r"""
            Function to initialize parameters from given A0, B0, C0 triplet initial weights.

            Args:
                A0 (Tensor): Initial state matrix of shape (nx, nx)
                B0 (Tensor): Initial input matrix of shape (nx, nu)
                C0 (Tensor): Initial output matrix of shape (ny, nx)
                gamma (float): Prescribed L2 gain
                alpha (float): Alpha stability margin, default is 0.0
            Raises:
                ValueError: If the parameterization method is 'riccati' and alpha is not 0.0.
                ValueError: If the LMI problem is infeasible with the prescribed gamma.
            Returns:
                None: The function initializes the parameters Q, P, S, G, H, and gmma_lmi.
        """

        if self.param == 'sqrtm':
            Q, P, S, G, H, M, gmma_lmi = self.submersion_inv_lmi(A0, B0, C0, gamma, epsilon=self.eps)
            self.Q = Q
            self.H = H
            self.M = M
        else:
            assert alpha == 0.0, "Alpha must be 0.0 for Riccati parameterization"
            print(f"Epsilon self.eps value {self.eps}")
            P, S, G = self.submersion_inv_riccati(A0, B0, C0, gamma, epsilon=self.eps)
            self.B = Parameter(B0)
            gmma_lmi = gamma

        self.P = P
        self.S = S
        self.G = Parameter(G)
        self.alpha = alpha
        self.gamma = gmma_lmi


    def submersion_inv_lmi(self, A: Tensor, B: Tensor, C: Tensor, gamma: float, epsilon=1e-4, solver="MOSEK"):
        """
            This function computes the parameters Q, P, S, G and H by solving the LMI problem.
            
            Args:
                A (Tensor): State matrix of shape (nx, nx)
                B (Tensor): Input matrix of shape (nx, nu)
                C (Tensor): Output matrix of shape (ny, nx)
                gamma (float): L2 gain
                epsilon (float): Small positive number for numerical stability
            
            Returns:
                Q (Tensor): Solution to the LMI problem of shape (nx, nx)
                P (Tensor): Solution to the LMI problem of shape (nx, nx)
                S (Tensor): Skew-symmetric matrix of shape (nx, nx)
                G (Tensor): Matrix of shape (ny, nx)
                H (Tensor): Matrix of shape (nx, nu) with orthogonal columns
                gamma_sys (float): Minimum gamma found in the LMI problem
        """
        with torch.no_grad():
            M, gamma_sys, P = HInfCont.solve(A, B, C, torch.zeros(self.ny, self.nu), alpha=0.0, tol=epsilon, solver=solver)
            
            if gamma_sys > gamma:
                raise ValueError(f"Infeasible problem with prescribed gamma : {gamma} min value = {gamma_sys}")
            else:
                self.gamma = gamma_sys  # Assign lowest gamma found if it's higher than the one prescribed
                

            print(f"M value : \n {M}")
            Ms = schur(M, self.nx, self.nu, self.nu)
            
            P_inv = torch.inverse(P)
            G = C @ P_inv

            # SVD of B/gamma
            U, Sigma, Vh = torch.linalg.svd(B / gamma_sys, full_matrices=True)
            # build Q with epsilon regularization in nullspace of B^T
            d = torch.cat([Sigma**2, epsilon * torch.ones(A.shape[0] - B.shape[1])])
            Q = (U * d.unsqueeze(0)) @ U.T

            # compute H
            Q_sqrt = sqrtm(Q)
            Q_inv_sqrt = torch.inverse(Q_sqrt)
            H = (1.0 / gamma_sys) * Q_inv_sqrt @ B

            S = A @ P_inv + 0.5 * (Q + G.T @ G + P_inv @ Ms @ P_inv)
        return Q, P, S, G, H, Ms, gamma_sys

    def submersion_inv_riccati(self, A: Tensor, B: Tensor, C: Tensor, gamma: float, epsilon=1e-8):
            """
                This function computes the parameters P, G and S by solving the Riccati equation.

                Args:
                    A (Tensor): State matrix of shape (nx, nx)
                    B (Tensor): Input matrix of shape (nx, nu)
                    C (Tensor): Output matrix of shape (ny, nx)
                    gamma (float): L2 gain
                    epsilon (float): Small positive number for numerical stability
                Returns:
                    P (Tensor): Solution to the Riccati equation of shape (nx, nx)
                    G (Tensor): Matrix of shape (ny, nx)
                    S (Tensor): Skew-symmetric matrix of shape (nx, nx)
            """
            with torch.no_grad():
                Q = (1 / gamma**2) * (B @ B.T)
                P, _ = solve_riccati_torch(A, B, C, gamma)
                G = Tensor(C) @ torch.inverse(P)

                S = Tensor(A) @ torch.inverse(P) + 0.5 *(Q + G.T @ G)

            return P, S, G

    def init_weights_(self, A0: Tensor, B0: Tensor, C0: Tensor, gamma: float, alpha: float):
        self._right_inverse(A0, B0, C0, gamma, alpha)
       
    @classmethod
    def copy(cls, model):

        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            float(model.gamma),
            float(model.alpha),
            param=model.param,
            epsilon=model.eps
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return L2BoundedLinear.copy(self)

    def check(self) -> bool:
        A, B, C = self._frame()

        M, gamma_check, P = HInfCont.solve(A, B, C, torch.zeros(self.ny, self.nu), self.alpha, tol=1e-6)
        return is_positive_definite(M) and is_positive_definite(P) and gamma_check <= self.gamma
    


class ExoL2BoundedLinear(SSModel):
    r"""
        Create a linear continuous-time state-space model with a prescribed L2 gain 
        and alpha stability beween exogenous inputs d and the measured outputs y.
        math::   
            \dot{x} &= Ax + Bu + B_dd\\
            y &= Cx

        Attributes:
            nu (int): Number of inputs
            ny (int): Number of outputs
            nx (int): Number of states
            nd (int): Number of exogenous inputs
            gamma (float): L2 gain
            alpha (float): alpha stability margin
            param (str): Parameterization method ('sqrtm' or 'ricccati')
            epsilon (float): Small positive number for positive definiteness
    """
    def __init__(self, nu: int, ny: int, nx: int, nd: int, gamma: float, alpha: float = 0.0,
                 param: str = 'sqrtm', epsilon=1e-6) -> None:
        # Here we parameterize the model to have a prescribed L2 gain between disturbances and outputs.
        super(ExoL2BoundedLinear, self).__init__(nu, ny, nx,nd)



        assert param in ['sqrtm', 'riccati'], "param must be 'sqrtm' or 'riccati'"
        assert alpha >= 0.0, "alpha must be non-negative"
        if param == 'riccati':
            assert alpha == 0.0, "alpha must be 0.0 for Riccati parameterization"


        self.gamma = gamma
        self.Ix = torch.eye(nx)
        self.eps = epsilon
        self.param = param
        self.alpha = alpha

        self.P = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.G = Parameter(nn.init.xavier_normal_(torch.empty(self.ny, self.nx)))
        self.B = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nu)))


        # Register relevant manifolds
        geo.positive_definite(self, 'P')
        geo.skew_symmetric(self, 'S')

        if param == 'sqrtm':
            self.Q = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
            geo.positive_definite(self, 'Q')
            self.H = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nd)))
            geo.orthogonal(self, 'H')
            self.M = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
            geo.positive_definite(self, 'M')
        else:
            self.Bd = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nd)))

        self._frame_cache = FrameCacheManager()


        
    def __repr__(self):
        repr = super(ExoL2BoundedLinear, self).__repr__() + "\n"
        repr += f"self.gamma={self.gamma}, self.alpha={self.alpha}, param={self.param}, epsilon={self.eps}"
        return repr
    
    def __str__(self):
        return self.__repr__()
    

    def forward(self, u, x, d=None) -> tuple[Tensor, Tensor]:
        """
            Forward pass of the linear system with exogenous inputs.
            Args:
                u (Tensor): Input tensor of shape (batch_size, nu)
                x (Tensor): State tensor of shape (batch_size, nx)
                d (Tensor): Exogenous input tensor of shape (batch_size, nd)
            Returns:
                dx (Tensor): State derivative tensor of shape (batch_size, nx)
                y (Tensor): Output tensor of shape (batch_size, ny)
        """
        if d is None:
            raise ValueError("Exogenous input d must be provided for this model. If None, use L2BoundedLinear instead.")
        A, B, C, Bd = self._frame()
        dx = x @ A.T + u @ B.T + d @ Bd.T

        y = x @ C.T
        return dx, y
    
    def _frame(self) -> tuple[Tensor, ...]:
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        if self.param == 'sqrtm':
            A, Bd, C = self._frame_sqrt()
        else:
            A, Bd, C = self._frame_riccati()
        B = self.B

        if self._frame_cache.is_caching:    
            self._frame_cache.cache = (A, B, C, Bd)
        return A, B, C, Bd
    
    def _frame_sqrt(self) -> tuple[Tensor, ...]:
        
        A = (-0.5 * (self.Q + self.G.T @ self.G + self.eps * self.Ix) + self.S) @ self.P -0.5* torch.inverse(self.P) @ self.M - self.alpha * self.Ix
        B = (self.gamma) * sqrtm(self.Q) @  self.H # type: ignore
        C = self.G @ self.P

        return A, B, C
    
    def _frame_riccati(self) -> tuple[Tensor, ...]:
        Bd = self.Bd

        A = (-0.5 * (1/self.gamma**2 * (Bd @ Bd.T) + self.G.T @ self.G) + self.S) @ self.P - self.alpha * self.Ix
        C = self.G @ self.P
        
        return A, Bd, C
    
    def _right_inverse(self, A0: Tensor, Bd0: Tensor, C0: Tensor, gamma: float, alpha=0.0):
        r"""
            Function to initialize parameters from given A0, B0, C0 triplet initial weights.

            Args:
                A0 (Tensor): Initial state matrix of shape (nx, nx)
                Bd0 (Tensor): Initial input matrix of shape (nx, nd)
                C0 (Tensor): Initial output matrix of shape (ny, nx)
                gamma (float): Prescribed L2 gain
                alpha (float): Alpha stability margin, default is 0.0
            Raises:
                ValueError: If the parameterization method is 'riccati' and alpha is not 0.0.
                ValueError: If the LMI problem is infeasible with the prescribed gamma.
            Returns:
                None: The function initializes the parameters Q, P, S, G, H, and gmma_lmi.
        """

        if self.param == 'sqrtm':
            Q, P, S, G, H, M, gmma_lmi = self.submersion_inv_lmi(A0, Bd0, C0, gamma, epsilon=self.eps)
            self.Q = Q
            self.H = H
            self.M = M
        else:
            assert alpha == 0.0, "Alpha must be 0.0 for Riccati parameterization"
            print(f"Epsilon self.eps value {self.eps}")
            P, S, G = self.submersion_inv_riccati(A0, Bd0, C0, gamma, epsilon=self.eps)
            self.Bd = Parameter(Bd0)
            gmma_lmi = gamma

        self.P = P
        self.S = S
        self.G = Parameter(G)
        self.alpha = alpha
        self.gamma = gmma_lmi


    def submersion_inv_lmi(self, A: Tensor, B: Tensor, C: Tensor, gamma: float, epsilon=1e-4, solver="MOSEK"):
        """
            This function computes the parameters Q, P, S, G and H by solving the LMI problem.
            
            Args:
                A (Tensor): State matrix of shape (nx, nx)
                B (Tensor): Input matrix of shape (nx, nu)
                C (Tensor): Output matrix of shape (ny, nx)
                gamma (float): L2 gain
                epsilon (float): Small positive number for numerical stability
            
            Returns:
                Q (Tensor): Solution to the LMI problem of shape (nx, nx)
                P (Tensor): Solution to the LMI problem of shape (nx, nx)
                S (Tensor): Skew-symmetric matrix of shape (nx, nx)
                G (Tensor): Matrix of shape (ny, nx)
                H (Tensor): Matrix of shape (nx, nu) with orthogonal columns
                gamma_sys (float): Minimum gamma found in the LMI problem
        """
        with torch.no_grad():
            M, gamma_sys, P = HInfCont.solve(A, B, C, torch.zeros(self.ny, self.nu), alpha=0.0, tol=epsilon, solver=solver)
            
            if gamma_sys > gamma:
                raise ValueError(f"Infeasible problem with prescribed gamma : {gamma} min value = {gamma_sys}")
            else:
                self.gamma = gamma_sys  # Assign lowest gamma found if it's higher than the one prescribed
                

            print(f"M value : \n {M}")
            Ms = schur(M, self.nx, self.nd, self.nd)
            
            
            P_inv = torch.inverse(P)
            G = C @ P_inv

            # SVD of B/gamma
            U, Sigma, Vh = torch.linalg.svd(B / gamma_sys, full_matrices=True)
            # build Q with epsilon regularization in nullspace of B^T
            d = torch.cat([Sigma**2, epsilon * torch.ones(A.shape[0] - B.shape[1])])
            Q = (U * d.unsqueeze(0)) @ U.T

            # compute H
            Q_sqrt = sqrtm(Q)
            Q_inv_sqrt = torch.inverse(Q_sqrt)
            H = (1.0 / gamma_sys) * Q_inv_sqrt @ B

            S = A @ P_inv + 0.5 * (Q + G.T @ G + P_inv @ Ms @ P_inv)
        return Q, P, S, G, H, Ms, gamma_sys

    def submersion_inv_riccati(self, A: Tensor, B: Tensor, C: Tensor, gamma: float, epsilon=1e-8):
            """
                This function computes the parameters P, G and S by solving the Riccati equation.

                Args:
                    A (Tensor): State matrix of shape (nx, nx)
                    B (Tensor): Input matrix of shape (nx, nu)
                    C (Tensor): Output matrix of shape (ny, nx)
                    gamma (float): L2 gain
                    epsilon (float): Small positive number for numerical stability
                Returns:
                    P (Tensor): Solution to the Riccati equation of shape (nx, nx)
                    G (Tensor): Matrix of shape (ny, nx)
                    S (Tensor): Skew-symmetric matrix of shape (nx, nx)
            """
            with torch.no_grad():
                Q = (1 / gamma**2) * (B @ B.T)
                P, _ = solve_riccati_torch(A, B, C, gamma)
                G = Tensor(C) @ torch.inverse(P)

                S = Tensor(A) @ torch.inverse(P) + 0.5 *(Q + G.T @ G)

            return P, S, G


    def init_weights_(self, A0: Tensor, B0: Tensor, C0: Tensor, G0: Tensor, gamma: float, alpha: float):
        self._right_inverse(A0, G0, C0, gamma, alpha)  # Initialize parameter's space for the disturbance model
        self.B = Parameter(B0)  # Set the input matrix B
       
    @classmethod
    def copy(cls, model):

        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            model.nd,
            float(model.gamma),
            float(model.alpha),
            param=model.param,
            epsilon=model.eps
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return ExoL2BoundedLinear.copy(self)

    def check(self) -> bool:
        A, _, C, G = self._frame()

        M, gamma_check, P = HInfCont.solve(A, G, C, torch.zeros(self.ny, self.nu), self.alpha, tol=1e-6)
        return is_positive_definite(M) and is_positive_definite(P) and gamma_check <= self.gamma