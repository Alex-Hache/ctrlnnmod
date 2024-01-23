'''
    This module include some bounded Lipschitz layers to use
    See https://github.com/acfr/LBDN for more details
'''
import torch
from torch.nn import Linear, ReLU
from torch.nn.parameter import Parameter
from linalg.utils import cayley
import torch.nn.functional as F


class SandwichLin(Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            if self.weight.norm() != 0:
                self.Q = cayley(self.alpha * self.weight / self.weight.norm())
            else:
                self.Q = cayley(self.alpha*self.weight)
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class SandwichFc(Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, actF=ReLU()):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None
        self.actF = actF

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(x * self.scale, Q[:, fout:])  # B*h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = self.actF(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x


class SandwichFcScaled(SandwichFc):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, actF=ReLU()):
        super().__init__(in_features, out_features, bias, scale, actF)
