import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from ctrlnmod.linalg.utils import sqrtm, check_observability, SoftmaxEta, InvSoftmaxEta
from ctrlnmod.utils import FrameCacheManager
from ctrlnmod.lmis.h2 import H2Cont
from ..linear import SSLinear, ExoSSLinear


class H2Linear(SSLinear):
    r"""
        Create a linear continuous-time state-space model with a prescribed H2 norm.
        math:: 
            \dot{x} &= Ax + Bu \\
            y &= Cx

        
        Attributes:
            nu (int): Number of inputs.
            ny (int): Number of outputs.
            nx (int): Number of states.
            gamma_2 (float): Prescribed H2 norm.
    """
    def __init__(self, nu: int, ny: int, nx: int, gamma_2: float) -> None:
        """
            Initialize the H2Linear model.
            Args:
                nu (int): Number of inputs.
                ny (int): Number of outputs.
                nx (int): Number of states.
                gamma_2 (float): Prescribed H2 norm.
        """
        super(H2Linear, self).__init__(nu, ny, nx)
        self.gamma2 = Tensor([gamma_2])

        self.C = Parameter(nn.init.xavier_normal_(torch.empty(self.ny, self.nx)))
        self.Wo_sqrt_inv = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.L = Parameter(torch.ones(self.nu))
        self.U = Parameter(nn.init.orthogonal_(torch.empty(self.nx, self.nu)))
        self.Vt = Parameter(nn.init.orthogonal_(torch.empty(self.nu, self.nu)))

        self.func_proj = SoftmaxEta(float(self.gamma2**2)) # type: ignore
        self.func_proj_inv = InvSoftmaxEta(float(self.gamma2**2)) # type: ignore


        # Register relevant manifolds
        geo.positive_definite(self, 'Wo_sqrt_inv')
        geo.skew_symmetric(self, 'S')
        geo.orthogonal(self,'U')
        geo.orthogonal(self, 'Vt')

        # Adding cache manager
        self._frame_cache = FrameCacheManager()

    def __repr__(self):
        rep = super(H2Linear, self).__repr__()
        return rep + f", gamma2 = {self.gamma2.item()}"
    
    def __str__(self):  
        return self.__repr__()

    def forward(self, u, x, d=None):
        A, B, C = self._frame()

        dx = x @ A.T + u @ B.T
        y = x @ C.T
        return dx, y

    def _frame(self, tol=1e-6) -> tuple[Tensor, ...]:
        """
            This function is the framing function from Parameter space to weights space.
        """

        # Si la mise en cache est active et qu'un cache existe, retourner le cache
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        sigma = torch.sqrt(self.func_proj(self.L))  # type: ignore
        

        B = self.Wo_sqrt_inv @ self.U @ torch.diag_embed(sigma) @ self.Vt

        Q = self.C.T @ self.C
        A = self.Wo_sqrt_inv @ self.Wo_sqrt_inv @ (-0.5 * Q + self.S)
        C = self.C
        
        # caching if necessary
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, B, C)

        return A, B, C

    @classmethod
    def copy(cls, model: 'H2Linear') -> 'H2Linear':
        '''
            This class method returns a copy of a given H2bounded model.

            We have to do this trick since self is not usable due to geotorch.
            Indeed when an object has parameterized attributes its class changes.
        '''
        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            float(model.gamma2)
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return H2Linear.copy(self)

    def check_(self, epsilon=1e-6):
        """
            Check if the model has the correct H2 norm.
            Args:
                epsilon (float): Tolerance for the check.
            Returns:
                bool: True if the model is valid, False otherwise.  
                float: The H2 norm of the model.
        """
        Wo = torch.inverse(self.Wo_sqrt_inv)@torch.inverse(self.Wo_sqrt_inv)
        A, B, C = self._frame()

        lyap = A.T @ Wo + Wo @ A
        dLyap = torch.dist(lyap, -C.T @ C)
        print(f"Distance to Lyapunov equation : {dLyap}")

        gamma_gram = torch.sqrt(torch.trace(B.T @ Wo @ B))
        dTrace = torch.dist(gamma_gram, self.gamma2)
        print(f"Traces : gram = {gamma_gram}  gamma2 : {self.gamma2} -- dist = {dTrace}")

        return bool((dLyap < epsilon) and (dTrace < epsilon)), gamma_gram

    def init_weights_(self, A0, B0, C0, requires_grad=True):
        """
            Initialize the model with given weights.
            Args:
                A0 (Tensor): Initial state matrix.
                B0 (Tensor): Initial input matrix.
                C0 (Tensor): Initial output matrix.
                requires_grad (bool): If True, gradients will be computed.
                """
        assert A0.shape == (self.nx, self.nx), f"A0 must be of shape {self.nx, self.nx}"
        assert B0.shape == (self.nx, self.nu), f"B0 must be of shape {self.nx, self.nu}"
        assert C0.shape == (self.ny, self.nx), f"C0 must be of shape {self.ny, self.nx}"
        is_observable = check_observability(A0, C0)
        if not is_observable:
            raise ValueError("The system is not observable with the given A0 and C0 matrices.")
        self.right_inverse_(A0, B0, C0, requires_grad)
    
    def right_inverse_(self, A, B, C, gamma2, check=False):
        """
            Method to initialize Parameter space from given weights

        """
        Wo_sqrt_inv, S, sigma, C, U, Vt = self.submersion_inv(A, B, C, gamma2=gamma2, check=check)
        self.Wo_sqrt_inv = Wo_sqrt_inv
        self.S = S
        self.U = U
        self.func_proj = SoftmaxEta(float(self.gamma2**2)) # type: ignore
        self.func_proj_inv = InvSoftmaxEta(float(self.gamma2**2))
        self.L = Parameter(self.func_proj_inv(Parameter(sigma)))
        self.Vt = Vt
        self.C = Parameter(C)

    def submersion_inv(self, A, B, C, gamma2=None, epsilon=1e-7, solver="MOSEK", check=False):
        """
            Inverse function from Weights space to parameter space.

            The LMI to obtain the observability grammian of the system is solved using CVXPY.
            Args:
                A (Tensor): State matrix.
                B (Tensor): Input matrix.
                C (Tensor): Output matrix.
                gamma2 (float): Prescribed H2 norm.
                epsilon (float): Tolerance for the check.
                solver (str): Solver to use for the LMI problem.
                check (bool): If True, check the feasibility of the problem.
            Returns:
                Wo_inv (Tensor): Inverse of the observability grammian.
                S (Tensor): Skew-symetric matrix.
                M (Tensor): Prescrjbed trace matrix.
                C (Tensor): Output matrix.
            Raises:
                ValueError: If the LMI problem is infeasible or unbounded.

        """
        with torch.no_grad():
            M, gamma2_sys, P = H2Cont.solve(A, B, C, solver=solver, epsilon=epsilon)  # type: ignore

            if gamma2 is None:
                gamma2 = 0.0

            if gamma2_sys > gamma2 and check:
                raise ValueError(f"Infeasible problem with prescribed gamma : {gamma2} min value = {gamma2_sys}")
            else:
                if gamma2_sys > gamma2:
                    print(
                        "Not in manifold with gamma2 = {} \n New gamma2 value assigned : g = {}".format(
                            gamma2, gamma2_sys
                        )
                    )
                self.gamma2 = Tensor([gamma2_sys])  # Assign lowest gamma found if it's higher than the one prescribed

            # Now initialize
            Wo = Tensor(P)
            # Applying square root first to improve conditioning number
            Wo_sqrt = sqrtm(Wo)
            cond_Wo_sqrt = torch.linalg.cond(Wo_sqrt)
            if cond_Wo_sqrt > 1e6:
                raise Warning(f"Condition number of Wo_sqrt is too high: {cond_Wo_sqrt}. The system may be ill-conditioned.")
            
            Wo_sqrt_inv = torch.inverse(Wo_sqrt)  # type: ignore
            Q = Tensor(-M[:self.nx, :self.nx])
            S = Wo @ Tensor(A) + 0.5 * Q
            U, sigma, Vt = torch.linalg.svd(Wo_sqrt@B, full_matrices=False)
        return Wo_sqrt_inv, S, sigma**2, Tensor(C), Tensor(U), Tensor(Vt)




class ExoH2Linear(ExoSSLinear):
    r""" 
        Create a linear continuous-time state-space model with a prescribed H2 norm and disturbance.
        math:: 
            \dot{x} &= Ax + Bu + Gd \\
            y &= Cx + Du
        
        Attributes:
            nu (int): Number of inputs.
            ny (int): Number of outputs.
            nx (int): Number of states.
            nd (int) number of exogenous signals
            gamma_2 (float): Prescribed H2 norm.

    """
    def __init__(self, nu: int, ny: int, nx: int, nd: int, gamma_2: float) -> None:
        """
            Initialize the ExoH2Linear model.
            Args:
                nu (int): Number of inputs.
                ny (int): Number of outputs.
                nx (int): Number of states.
                nd (int): Number of exogenous signals
                gamma_2 (float): Prescribed H2 norm.
        """
        super(ExoH2Linear, self).__init__(nu, ny, nx, nd)
    
        self.gamma2 = Tensor([gamma_2])
        self.Wo_sqrt_inv = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.L = Parameter(torch.ones(self.nd))
        self.U = Parameter(nn.init.orthogonal_(torch.empty(self.nx, self.nd)))
        self.Vt = Parameter(nn.init.orthogonal_(torch.empty(self.nd, self.nd)))

        self.func_proj = SoftmaxEta(float(self.gamma2**2)) # type: ignore
        self.func_proj_inv = InvSoftmaxEta(float(self.gamma2**2)) # type: ignore


        # Register relevant manifolds
        geo.positive_definite(self, 'Wo_sqrt_inv')
        geo.skew_symmetric(self, 'S')
        geo.orthogonal(self,'U')
        geo.orthogonal(self, 'Vt')

    def __repr__(self):
        rep = super(ExoH2Linear, self).__repr__()
        return rep + f", gamma2 = {self.gamma2.item()}"
    
    def __str__(self):  
        return self.__repr__()

    def forward(self, u, x, d):
        A, B, C, G = self._frame()

        dx = x @ A.T + u @ B.T + d @ G.T
        y = x @ C.T
        return dx, y
    

    def _frame(self, tol=1e-8) -> tuple[Tensor, ...]:
        """
            This function is the framing function from Parameter space to weights space.
        """

        # Si la mise en cache est active et qu'un cache existe, retourner le cache
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        sigma = torch.sqrt(self.func_proj(self.L))  # type: ignore
        

        G = self.Wo_sqrt_inv @ self.U @ torch.diag_embed(sigma) @ self.Vt

        Q = self.C.weight.T @ self.C.weight
        A = self.Wo_sqrt_inv@self.Wo_sqrt_inv@(-0.5 * Q + self.S)
        
        # caching if necessary
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, self.B.weight, self.C.weight, G)

        return A, self.B.weight, self.C.weight, G
    

    def init_weights_(self, A0, B0, C0, G0, gamma2=None):
            """
                Initialize the model with given weights.
                Args:
                    A0 (Tensor): Initial state matrix.
                    B0 (Tensor): Initial input matrix.
                    C0 (Tensor): Initial output matrix.
                    G0 (Tensor): Initial disturbance matrix.
                    requires_grad (bool): If True, gradients will be computed.
                    """
            assert A0.shape == (self.nx, self.nx), f"A0 must be of shape {self.nx, self.nx}"
            assert B0.shape == (self.nx, self.nu), f"B0 must be of shape {self.nx, self.nu}"
            assert C0.shape == (self.ny, self.nx), f"C0 must be of shape {self.ny, self.nx}"
            assert G0.shape == (self.nx, self.nd), f"G0 must be of shape {self.nx, self.nd}"

            is_observable = check_observability(A0, C0)
            if not is_observable:
                raise ValueError("The system is not observable with the given A0 and C0 matrices.")
            self.right_inverse_(A0, B0, C0, G0, gamma2)
        
    def right_inverse_(self, A, B, C, G, gamma2, check=False):
        """
            Method to initialize Parameter space from given weights

        """
        Wo_sqrt_inv, S, sigma, C, U, Vt = self.submersion_inv(A, G, C, gamma2=gamma2, check=check)
        self.Wo_sqrt_inv = Wo_sqrt_inv
        self.S = S
        self.U = U
        self.func_proj = SoftmaxEta(float(self.gamma2**2)) # type: ignore
        self.func_proj_inv = InvSoftmaxEta(float(self.gamma2**2))
        self.L = Parameter(self.func_proj_inv(Parameter(sigma)))
        self.Vt = Vt
        self.C.weight = Parameter(C)
        self.B.weight = Parameter(B)

    def submersion_inv(self, A, G, C, gamma2=None, epsilon=1e-5, solver="MOSEK", check=False):
        """
            Inverse function from Weights space to parameter space.

            The LMI to obtain the observability grammian of the system is solved using CVXPY.
            Args:
                A (Tensor): State matrix.
                G (Tensor): Input disturbance matrix.
                C (Tensor): Output matrix.
                gamma2 (float): Prescribed H2 norm.
                epsilon (float): Tolerance for the check.
                solver (str): Solver to use for the LMI problem.
                check (bool): If True, check the feasibility of the problem.
            Returns:
                Wo_inv (Tensor): Inverse of the observability grammian.
                S (Tensor): Skew-symetric matrix.
                Sigma (Tensor): Prescribed trace vector.
                C (Tensor): Output matrix.
                U (Tensor): Orthogonal matrix of size nx * nx.
                Vt (Tensor): Orthogonal matrix of size nd * nd.
            Raises:
                ValueError: If the LMI problem is infeasible or unbounded.

        """
        with torch.no_grad():
            M, gamma2_sys, P = H2Cont.solve(A, G, C, solver=solver, epsilon=epsilon)  # type: ignore

            if gamma2 is None:
                gamma2 = 0.0

            if gamma2_sys > gamma2 and check:
                raise ValueError(f"Infeasible problem with prescribed gamma : {gamma2} min value = {gamma2_sys}")
            else:
                if gamma2_sys > gamma2:
                    print(
                        "Not in manifold with gamma2 = {} \n New gamma2 value assigned : g = {}".format(
                            gamma2, gamma2_sys
                        )
                    )
                self.gamma2 = Tensor([gamma2_sys])  # Assign lowest gamma found if it's higher than the one prescribed


            # Now initialize
            Wo = Tensor(P)
            # Applying square root first to improve conditioning number
            Wo_sqrt = sqrtm(Wo)
            cond_Wo_sqrt = torch.linalg.cond(Wo_sqrt)
            if cond_Wo_sqrt > 1e6:
                raise Warning(f"Condition number of Wo_sqrt is too high: {cond_Wo_sqrt}. The system may be ill-conditioned.")
            
            Wo_sqrt_inv = torch.inverse(Wo_sqrt)  # type: ignore
            Q = Tensor(-M[:self.nx, :self.nx])
            S = Wo @ Tensor(A) + 0.5 * Q

            U, sigma, Vt = torch.linalg.svd(Wo_sqrt@G, full_matrices=False)
        return Wo_sqrt_inv, S, sigma**2, Tensor(C), Tensor(U), Tensor(Vt)
    
    @classmethod
    def copy(cls, model: 'ExoH2Linear') -> 'ExoH2Linear':
        '''
            This class method returns a copy of a given H2bounded model.

            We have to do this trick since self is not usable due to geotorch.
            Indeed when an object has parameterized attributes its class changes.
        '''
        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            model.nd,
            float(model.gamma2)
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return ExoH2Linear.copy(self)
