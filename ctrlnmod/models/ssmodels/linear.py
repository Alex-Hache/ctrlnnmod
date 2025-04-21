from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from geotorch_custom.parametrize import is_parametrized
import torch
from torch import Tensor
from ctrlnmod.utils import FrameCacheManager
from .base import SSModel
from typing import Dict, Tuple, Optional, Union, Any


class SSLinear(SSModel):
    """
    Module corresponding to a continuous-time linear state-space model
        With or without disturbances
        \dot{x} = Ax + Bu + (Gd)
        y = Cx

    Attributes : 
        nu (int): number of inputs
        ny (int): number of outputs
        nx (int): number of states
        nd (int): number of disturbances
            Defaults to 0 (no exogenous inputs)
        alpha (float, optional) : upper bound on decay rate of matrix A (necessarily Hurwitz)
    """

    def __init__(
        self, input_dim: int, output_dim: int, state_dim: int, dist_dim: int = 0,
        alpha: Optional[float] = None) -> None:
        super(SSLinear, self).__init__(input_dim, output_dim, state_dim)

        self.nu = input_dim
        self.nx = state_dim
        self.ny = output_dim
        self.nd = dist_dim
        self.A = Linear(self.nx, self.nx, bias=False)
        self.B = Linear(self.nu, self.nx, bias=False)
        self.C = Linear(self.nx, self.ny, bias=False)

        self.alpha = alpha
        if self.nd > 0:
            self.G = Linear(self.nd, self.nx, bias=False)

        # Is A alpha stable ?
        if self.alpha is not None:
            geo.alpha_stable(self.A, 'weight', alpha=alpha)

        self._frame_cache = FrameCacheManager()


    def __repr__(self):
        if is_parametrized(self.A):
            return "Stable_Linear_ss" + f"_{self.alpha}"
        else:
            return "Linear_ss"

    def forward(self, u, x, d=None):
        dx = self.A(x) + self.B(u)

        if self.nd > 0 and d is not None:
            dx += self.G(d)
    
        y = self.C(x)

        return dx, y

    def _frame(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache

        A, B, C = self.A.weight, self.B.weight, self.C.weight
        if self.nd > 0:
            G = self.G.weight
            # Store in the cache if it is active and not yet set up
            if self._frame_cache.is_caching:
                self._frame_cache.cache = (A, B, C, G)
            return A, B, C, G
        else:
            if self._frame_cache.is_caching:
                self._frame_cache.cache = (A, B, C)
            return A, B, C

    def eval_(self):
        if self.nd > 0:
            return self.A.weight, self.B.weight, self.C.weight, self.G.weight
        else:
            return self.A.weight, self.B.weight, self.C.weight

    def right_inverse_(self, A0, B0, C0, requires_grad=True, G0=None):
        # Check matrix dimensions
        assert self.nx == A0.shape[0] == A0.shape[
            1], f"Given A matrix has incorrect size: nx = {self.nx}, found {A0.shape}"
        assert (
            self.nx, self.nu) == B0.shape, f"Given B matrix has incorrect size: expected ({self.nx}, {self.nu}), found {B0.shape}"
        assert (
            self.ny, self.nx) == C0.shape, f"Given C matrix has incorrect size: expected ({self.ny}, {self.nx}), found {C0.shape}"

        # Initialize A
        if is_parametrized(self.A):
            self.A.weight = A0
        else:
            self.A.weight = Parameter(A0)

        # Initialize B and C
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)

        self.A0 = self.A.weight.detach().clone()
        self.B0 = self.B.weight.detach().clone()
        self.C0 = self.C.weight.detach().clone()

        if self.nd > 0:
            self.G.weight = Parameter(G0)
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
            if self.nd > 0:
                self.G.requires_grad_(False)

    def check_(self):
        if self.alpha is None:
            alpha = 0.0
        else:
            alpha = self.alpha
        eig_vals = torch.real(torch.linalg.eigvals(self.A.weight))
        return torch.all(eig_vals <= alpha), torch.max(eig_vals)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def init_weights_(self, A0, B0, C0, requires_grad=True, G0=None, margin=0.1, adjust_alpha=False):
        # Compute the Lyapunov exponent of A0
        A0_lyap_exp = -torch.max(torch.real(torch.linalg.eigvals(A0)))

        if adjust_alpha:  # For now to be set to fals
            # Adjust alpha to be slightly smaller than A0's Lyapunov exponent
            self.alpha = torch.tensor(
                float(A0_lyap_exp) - margin, device=A0.device)
        else:
            # Shift eigenvalues of A0 to match the desired alpha
            if self.alpha is None:
                alpha = 0.0
            else:
                alpha = self.alpha
            if A0_lyap_exp < alpha:
                shift = self.alpha - A0_lyap_exp + margin
                # Move to left part of the left half plane
                A0 = A0 - shift * torch.eye(self.nx, device=A0.device)
        if self.nd > 0 and G0 is not None:
            self.right_inverse_(A0, B0, C0, requires_grad=requires_grad, G0=G0)
        else:
            self.right_inverse_(A0, B0, C0, requires_grad=requires_grad)