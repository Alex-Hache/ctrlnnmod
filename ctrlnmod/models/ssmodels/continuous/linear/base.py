from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from geotorch_custom.parametrize import is_parametrized
import torch
from torch import Tensor
from ctrlnmod.utils import FrameCacheManager
from ...base import SSModel
from typing import Dict, Tuple, Optional, Union, Any
from ctrlnmod.linalg.utils import get_lyap_exp

class SSLinear(SSModel):
    r"""
    A continuous-time linear state-space model without exogenous inputs and direct trerm :
    .. math::    
        \dot{x} &= Ax + Bu \\
        y &= Cx

    Attributes : 
        nu (int): number of inputs
        ny (int): number of outputs
        nx (int): number of states
        alpha (float, optional) : upper bound on decay rate of matrix A (necessarily Hurwitz)

    Todo:
        - Add support for direct term
    """

    def __init__(
        self, input_dim: int, output_dim: int, state_dim: int, alpha: Optional[float] = None) -> None:
        """
        Args: 
            input_dim (int): number of inputs
            output_dim (int): number of outputs
            state_dim (int): number of states
            alpha (float, optional): upper bound on decay rate of matrix A (necessarily Hurwitz)
        """
        super(SSLinear, self).__init__(input_dim, output_dim, state_dim)

        self.A = Linear(self.nx, self.nx, bias=False)
        self.B = Linear(self.nu, self.nx, bias=False)
        self.C = Linear(self.nx, self.ny, bias=False)

        self.alpha = alpha

        # Is A alpha stable ?
        if alpha is not None:
            geo.alpha_stable(self.A, 'weight', alpha=alpha)

        self._frame_cache = FrameCacheManager()


    def __repr__(self):
        rep = super(SSLinear, self).__repr__()
        if self.alpha is not None:
            rep += f"_{self.alpha}"
        return rep
    
    def __str__(self):  
        return self.__repr__()

    def forward(self, u, x, d= None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            u (Tensor): input tensor of shape (batch_size, nu)
            x (Tensor): state tensor of shape (batch_size, nx)
        Returns:
            dx (Tensor): state derivative tensor of shape (batch_size, nx)
            y (Tensor): output tensor of shape (batch_size, ny)
        """
        dx = self.A(x) + self.B(u)
        y = self.C(x)
        return dx, y

    def _frame(self) -> tuple[Tensor, ...]:
        """
            This method is used to compute the weights of the model from variables in the parameter space.
        Returns:
            tuple: A, B, C
        """
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache

        A, B, C = self.A.weight, self.B.weight, self.C.weight
        # Store in the cache if it is active and not yet set up
        if self._frame_cache.is_caching:
                self._frame_cache.cache = (A, B, C)
        return A, B, C

    def _right_inverse(self, A0, B0, C0, requires_grad=True) -> None:
        """
            This method is a right inverse of the _frame method : 
            it initalizes the parameter space with suitable vlaues in the weights space.

        Args:
            A0 (Tensor): A matrix of shape (nx, nx)
            B0 (Tensor): B matrix of shape (nx, nu)
            C0 (Tensor): C matrix of shape (ny, nx)
            requires_grad (bool): if True, the parameters will be updated during training
        """

        # Initialize A
        if is_parametrized(self.A):
            print(f"A before init : {self.A.weight}")
            self.A.weight = A0
            print(f"A after init : {self.A.weight}")
        else:
            self.A.weight = Parameter(A0)

        # Initialize B and C
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)

        self.A0 = self.A.weight.detach().clone()
        self.B0 = self.B.weight.detach().clone()
        self.C0 = self.C.weight.detach().clone()

        # Set requires_grad
        if not requires_grad:
            if is_parametrized(self.A):
                for parameter in self.A.parameters():
                    parameter.requires_grad_(False)
            else:
                self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def to_hinf(self):
        A, B, C = self._frame()
        return A, B, C
    
    def check_(self):
        """Check if the system is stable by checking the eigenvalues of A.

        Returns:
            bool: True if the system is stable, False otherwise
            float: maximum eigenvalue of A
        """

        if self.alpha is None:
            alpha = 0.0
        else:
            alpha = self.alpha
        eig_vals = torch.real(torch.linalg.eigvals(self.A.weight))
        return bool(torch.all(eig_vals <= alpha)), torch.max(eig_vals)

    def clone(self):
        """
        Clone the model. This method is called by the simulator to create a copy of the model.
        Returns:
            SSLinear: A copy of the model
        
        """  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def init_weights_(self, A0: Tensor, B0: Tensor, C0: Tensor, requires_grad=True):
        """Initialize the weights of the model.
        Args:
            A0 (Tensor): A matrix of shape (nx, nx)
            B0 (Tensor): B matrix of shape (nx, nu)
            C0 (Tensor): C matrix of shape (ny, nx)
            requires_grad (bool): if True, the parameters will be updated during training
        """
        # Check matrix dimensions
        assert self.nx == A0.shape[0] == A0.shape[
            1], f"Given A matrix has incorrect size: nx = {self.nx}, found {A0.shape}"
        assert (
            self.nx, self.nu) == B0.shape, f"Given B matrix has incorrect size: expected ({self.nx}, {self.nu}), found {B0.shape}"
        assert (
            self.ny, self.nx) == C0.shape, f"Given C matrix has incorrect size: expected ({self.ny}, {self.nx}), found {C0.shape}"
        # Compute the Lyapunov exponent of A0
        A0_lyap_exp = get_lyap_exp(A0)
        if self.alpha is not None:
            assert A0_lyap_exp > self.alpha, f"Lyapunov exponent of A0 ({A0_lyap_exp}) is not bigger than alpha ({self.alpha})"
        self._right_inverse(A0, B0, C0, requires_grad=requires_grad)


    
class ExoSSLinear(SSModel):
    r"""
    A continuous-time linear state-space model with exogenous inputs :
    math::
        \dot{x} &= Ax + Bu + Gd \\
        y &= Cx
    where :math:`d` is the exogenous input.    

    Attributes :
        nu (int): number of inputs
        ny (int): number of outputs
        nx (int): number of states
        nd (int): number of exogenous inputs
        alpha (float, optional) : upper bound on decay rate of matrix A (necessarily Hurwitz)
    """
    def __init__(
        self, input_dim: int, output_dim: int, state_dim: int, dist_dim: int,
        alpha: Optional[float] = None) -> None:
        """
        Args:
            input_dim (int): number of inputs
            output_dim (int): number of outputs
            state_dim (int): number of states
            dist_dim (int): number of exogenous inputs
            alpha (float, optional): upper bound on decay rate of matrix A (necessarily Hurwitz)
        """
        # Call the parent constructor
        # with the appropriate parameters
        # Note: dist_dim is not used in the parent constructor
        # but is used in the child class to initialize the G matrix
        super(ExoSSLinear, self).__init__(input_dim, output_dim, state_dim, dist_dim)
    
        self.A = Linear(self.nx, self.nx, bias=False)
        self.B = Linear(self.nu, self.nx, bias=False)
        self.C = Linear(self.nx, self.ny, bias=False)
        self.G = Linear(self.nd, self.nx, bias=False)

        self.alpha = alpha

        # Is A alpha stable ?
        if alpha is not None:
            geo.alpha_stable(self.A, 'weight', alpha=alpha)

        self._frame_cache = FrameCacheManager()

    def __repr__(self):
        """String representation of the model."""
        rep = super(ExoSSLinear, self).__repr__()
        return rep

    def forward(self, u, x, d=None) -> Tuple[Tensor, Tensor]:
        r"""
        
        Args:
            u (Tensor): input tensor of shape (batch_size, nu)
            x (Tensor): state tensor of shape (batch_size, nx)
            d (Tensor): exogenous input tensor of shape (batch_size, nd)
        Returns:
            dx (Tensor): state derivative tensor of shape (batch_size, nx)
            y (Tensor): output tensor of shape (batch_size, ny)
        """

        dx, y = self.A(x) + self.B(u) + self.G(d), self.C(x)
        return dx, y

    def _frame(self) -> tuple[Tensor, ...]:
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache

        A, B, C, G = self.A.weight, self.B.weight, self.C.weight, self.G.weight
        
        # Store in the cache if it is active and not yet set up
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, B, C, G)
        return A, B, C, G
      

    def eval_(self) -> Tuple[Tensor, ...]:
        return self.A.weight, self.B.weight, self.C.weight, self.G.weight

    def to_hinf(self):
        A, B, C, G = self._frame()
        return A, G, C
    
    def _right_inverse(self, A0, B0, C0, G0, requires_grad=True):
        """
            This method is a right inverse of the frame_ method : it initalize the parameter space with suitable vlaues in the weights space.
        Args:
            A0 (Tensor): A matrix of shape (nx, nx)
            B0 (Tensor): B matrix of shape (nx, nu)
            C0 (Tensor): C matrix of shape (ny, nx)
            G0 (Tensor): G matrix of shape (nx, nd)
            requires_grad (bool): if True, the parameters will be updated during training
        """
        # Initialize A
        if is_parametrized(self.A):
            self.A.weight = A0 # If A is parameterized we use geotorch mechanism to set the weight
        else:
            self.A.weight = Parameter(A0)

        # Initialize B and C
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)
        self.G.weight = Parameter(G0)


        # Store the initial weights
        self.A0 = self.A.weight.detach().clone()
        self.B0 = self.B.weight.detach().clone()
        self.C0 = self.C.weight.detach().clone()
        self.G0 = self.G.weight.detach().clone()

        # Set requires_grad
        if not requires_grad:
            if is_parametrized(self.A):
                for parameter in self.A.parameters():
                    parameter.requires_grad_(False)
            else:
                self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)
            self.G.requires_grad_(False)


    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx, self.nd, alpha=self.alpha)
        # Copy the parameters
        copy.load_state_dict(self.state_dict())
        return copy

    def init_weights_(self, A0: Tensor, B0: Tensor, C0: Tensor, G0: Tensor, requires_grad=True):
        """Initialize the weights of the model.
        Args:
            A0 (Tensor): A matrix of shape (nx, nx)
            B0 (Tensor): B matrix of shape (nx, nu)
            C0 (Tensor): C matrix of shape (ny, nx)
            G0 (Tensor): G matrix of shape (nx, nd)
            requires_grad (bool, optional): if True, the parameters will be updated during training
        """
        assert self.nx == A0.shape[0] == A0.shape[1]
        assert B0.shape == (self.nx, self.nu)
        assert C0.shape == (self.ny, self.nx)
        assert G0.shape == (self.nx, self.nd)

        A0_lyap_exp = get_lyap_exp(A0)
        if self.alpha is not None:
            assert A0_lyap_exp > self.alpha, f"Lyapunov exponent of A0 ({A0_lyap_exp}) is not bigger than alpha ({self.alpha})"
        self._right_inverse(A0, B0, C0, G0, requires_grad=requires_grad)
