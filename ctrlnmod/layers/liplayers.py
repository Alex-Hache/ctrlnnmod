'''
    This module include some bounded Lipschitz layers to use
    See https://github.com/acfr/LBDN for more details
'''
import torch
from torch.nn import Linear, ReLU
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import geotorch_custom as geo


class SandwichLinear(Linear):
    r"""
    A specific linear layer with Lipschitz constant equal to scale

    .. math::
        h_{out} = \sqrt{2} A^T \Psi \sigma \left( \sqrt{2} \Psi^{-1} B h_{in} + b \right)
    '''

    Attributes
    ----------
    alpha : Tensor
        scaling parameter for computation
    scale : float
        scaling parameter to define Lipschitz constant
    AB : bool
        If true the product of A and B matrices is computed
        instead of just B.
    param : str
        'expm' or 'cayley' way to parameterize the matrices on the Stiefel manifold
    scale : float
        the input tensor is multiplied by scale
    """
    def __init__(self, in_features, out_features, scale=1.0, param='expm', bias=True, AB=False):

        super().__init__(in_features + out_features, out_features, bias)
        self.scale = scale
        self.AB = AB
        self.param = param
        geo.orthogonal(self, 'weight', triv=param)  # let geotorch handle errors

    def forward(self, x):
        fan_out, _ = self.weight.shape
        x = F.linear(self.scale * x, self.weight[:, fan_out:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, self.weight[:, :fan_out].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class SandwichLayer(Linear):
    r"""
    A specific Sandwich layer with Lipschitz constant equal to scale

    .. math::
        h_{out} = \sqrt{2} A^T \Psi \sigma \left( \sqrt{2} \Psi^{-1} B h_{in} + b \right)
    '''

    Attributes
    ----------
    alpha : Tensor
        scaling parameter for computation
    scale : float
        scaling parameter to define Lipschitz constant
    AB : bool
        If true the product of A and B matrices is computed
        instead of just B.
    act_f :
        activation function for the sandwich layer
    param : str
        'expm' or 'cayley' way to parameterize the matrices on the Stiefel manifold
    scale : float
        the input tensor is multiplied by scale
    """

    def __init__(self, in_features, out_features, scale=1.0, act_f=ReLU(), param='expm', bias=True, AB=True):
        super().__init__(in_features + out_features, out_features, bias)
        self.param = param
        geo.orthogonal(self, 'weight', triv=param)  # let geotorch handle errors
        self.scale = scale
        self.psi = Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.act_f = act_f

    def forward(self, x):
        fan_out, _ = self.weight.shape
        x = F.linear(self.scale * x, self.weight[:, fan_out:])  # B @ x
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = self.act_f(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, self.weight[:, :fan_out].T)  # sqrt(2) A^top \Psi z
        return x
