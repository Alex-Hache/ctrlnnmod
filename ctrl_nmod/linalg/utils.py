import numpy as np
import scipy
import torch
from torch.autograd import Function


def isSDP(L):
    '''
    L is tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    isAllEigPos = torch.all(torch.real(eigval) > 0)
    isSymetric = torch.all(L == L.T)
    if not isAllEigPos:
        print("Not all eigenvalues are positive \n")
    if not isSymetric:
        print("Matrix is not symmetric \n")
    bSDP = isSymetric and isAllEigPos
    return bSDP


def getEigenvalues(L):
    '''
        params :
            - L pytorch Tensor
    '''
    return torch.linalg.eigvals(L)


# from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I_nin = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I_nin + A)
    return torch.cat((iIpA @ (I_nin - A), -2 * V @ iIpA), axis=1)  # type: ignore


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply
