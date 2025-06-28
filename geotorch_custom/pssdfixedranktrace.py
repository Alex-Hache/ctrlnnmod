import torch

from .symmetric import SymF


class SoftmaxEta(torch.nn.Module):
    def __init__(self, eta: float = 1.0, epsilon=1e-6) -> None:
        super(SoftmaxEta, self).__init__()
        self.eta = eta

    def __name__(self):
        return f"SoftmaxEta : eta = {self.eta}"

    def forward(self, x):
        return self.eta * torch.nn.functional.softmax(x, dim=0)


class InvSoftmaxEta(torch.nn.Module):
    def __init__(self, eta: float = 1.0, epsilon=1e-6) -> None:
        super(InvSoftmaxEta, self).__init__()
        self.eta = eta
        self.epsilon = epsilon

    def __name__(self):
        return f"SoftmaxEtaInv : eta = {self.eta}"

    def forward(self, s):
        x = torch.log(s / self.eta)
        return x


class PSSDFixedRankTrace(SymF):
    fs = {"softmax": (SoftmaxEta, InvSoftmaxEta)}

    def __init__(self, size, rank, trace, f, triv="expm"):
        self.trace = torch.Tensor([trace])
        super().__init__(size, rank, PSSDFixedRankTrace.parse_f(f, trace), triv)

    @staticmethod
    def parse_f(f, trace):
        if f == "softmax":
            func = SoftmaxEta(trace)
            inv = InvSoftmaxEta(trace)
            return (func, inv)
        elif f in PSSDFixedRankTrace.fs.keys():
            return PSSDFixedRankTrace.fs[f][0](trace), PSSDFixedRankTrace.fs[f][1](trace)
        elif callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable or a pair of callables. "
                "Should be one of {}. Found {}".format(list(PSSDFixedRankTrace.fs.keys()), f)
            )

    def in_manifold_eigen(self, L, eps=1e-6):
        r"""
        Checks that an ascending ordered vector of eigenvalues is in the manifold.

        Args:
            L (torch.Tensor): Vector of eigenvalues of shape `(*, rank)`
            eps (float): Optional. Threshold at which the eigenvalues are
                considered to be zero
                Default: ``1e-6``
        """
        bSym = super().in_manifold_eigen(L, eps)
        bZero = (L[..., -self.rank:] >= eps).all().item()
        bTrace = (torch.dist(L[..., -self.rank:].sum(), self.trace) <= 1e-4).item()
        if bSym and bZero and bTrace:
            return True
        else:
            return False

    def sample(self, init_=torch.nn.init.xavier_normal_, eps=5e-6):
        r'''
            Sample a matrix with given trace since its svd is in the image of
            f function defined earlier
        '''
        L, Q = super().sample(factorized=True, init_=init_)
        with torch.no_grad():
            L[L < eps] = eps
        return (Q * self.f(L).unsqueeze(-2)) @ Q.transpose(-2, 1)
