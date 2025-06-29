import torch.nn as nn
import torch.nn.functional as F
import geotorch_custom as geo
import torch
from torch.linalg import cholesky, inv
from torch import Tensor
from typing import List, Optional

class CustomSoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0, margin=0.01):
        super().__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)
        self.margin = margin

    def forward(self, x):
        return self.softplus(x) + self.margin  # ensures strictly positive output



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
        self,
        n_in: int,
        n_out: int,
        hidden_layers: List[int],
        act_f: nn.Module = nn.Tanh(),
        func: str = "softplus",
        tol: float = 0.01,
        scale: float = 1.0,
        use_residual: bool = False,
    ) -> None:
        r"""
        Constructs a matrix-valued function :math:`\beta(x` \in \mathbb{R}^{n_\text{out} \times n_\text{out}}`, defined as:

            :math:`\beta(x) = U · diag(\sigma(x)) · V^T`

        where:

        - :math:`\( U \in \mathbb{R}^{n_\text{out} \times n_\text{in}} \)` and :math:`\( V \in \mathbb{R}^{n_\text{in} \times n_\text{out}} \)` are orthogonal matrices.
        - :math:`\( \sigma(x) \in \mathbb{R}^{n_\text{out}} \)` is a vector of positive scalars produced by an MLP.
        - :math:`\( \text{diag}(\sigma(x)) \)` denotes a diagonal matrix with positive entries.

        Args:
            n_out (int): Output size of the layer. Defines the shape of the square matrix \beta(x).
            n_in (int): Input feature size (dimensionality of x).
            hidden_layers (List[int]): List of hidden layer sizes for the MLP computing \sigma(x).
            act_f (nn.Module): Activation function used between hidden layers.
            func (str): Type of positive function used at output ("softplus" or "relu").
            tol (float): Minimum margin to enforce positive definiteness.
            scale (float): Global scaling of the diagonal values.
            use_residual (bool): If True, adds an identity matrix to \beta(x) for residual stabilization.
        
        TODO:
            - Implement support for building beta lyer with Lipschitz bounded MLPs instead of standard MLPs.
        """
        super().__init__()

        self.n_out = n_out
        self.n_in = n_in
        self.scale = scale
        self.use_residual = use_residual

        # U and V: learnable orthogonal matrices
        self.U = nn.Linear(n_in, n_out, bias=False)
        self.V = nn.Linear(n_out, n_in, bias=False)
        nn.init.eye_(self.U.weight)
        nn.init.eye_(self.V.weight)

        geo.orthogonal(self.U, "weight")
        geo.orthogonal(self.V, "weight")

        # Build the MLP for sigma(x)
        layers = []
        in_dim = n_in
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_f)
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_out))
        self.mlp_sigma = nn.Sequential(*layers)

        # Choose output positivity function
        if func == "softplus":
            self.pos_func = CustomSoftplus(beta=1.0, threshold=20.0, margin=tol)
        elif func == "relu":
            self.pos_func = nn.ReLU()
        else:
            raise NotImplementedError(f"Positive function '{func}' not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the matrix-valued function \beta(x).

        Args:
            x (Tensor): Input tensor of shape (..., n_in)

        Returns:
            Tensor: Output tensor of shape (..., n_out, n_out)
        """
        sig = self.mlp_sigma(x)                        # (..., n_out)
        sig = self.pos_func(sig) / self.scale          # (..., n_out)
        sig = sig.unsqueeze(-1)                        # (..., n_out, 1)

        U_mat = self.U.weight                          # (n_in, n_out)
        Vt = self.V.weight.transpose(0, 1)             # (n_out, n_out)

        mid = sig * Vt                                 # (..., n_out, n_out)
        beta = U_mat @ mid.transpose(1,2)                             # (..., n_out, n_out)

        if self.use_residual:
            identity = torch.eye(self.n_out, device=x.device).expand_as(beta)
            beta = identity + beta

        return beta

    def clone(self) -> "BetaLayer":
        """
        Create a clone of the BetaLayer with the same parameters.
        """
        cloned_layer = BetaLayer(
            n_in=self.n_in,
            n_out=self.n_out,
            hidden_layers=[layer.out_features for layer in self.mlp_sigma if isinstance(layer, nn.Linear)],
            act_f=self.mlp_sigma[1],  # Assuming the first activation function is used
            func="softplus",  # Default to softplus, can be changed if needed
            tol=0.01,  # Default tolerance
            scale=self.scale,
            use_residual=self.use_residual,
        )
        cloned_layer.load_state_dict(self.state_dict())
        return cloned_layer

