import numpy as np
import scipy
import torch
from torch.autograd import Function
from torch.linalg import eigvals
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.vstack import vstack
from typing import Union


def block_diag(arr_list):
    '''create a block diagonal matrix from a list of cvxpy matrices'''

    # rows and cols of block diagonal matrix
    n = np.sum([arr.shape[1] for arr in arr_list])

    # loop to create the list for the bmat function
    block_list = []  # list for bmat function
    ind = np.array([0, 0])
    for arr in arr_list:
        # index of the end of arr in the block diagonal matrix
        ind += arr.shape

        # list of one row of blocks
        horz_list = [arr]

        # block of zeros to the left of arr
        zblock_l = np.zeros((arr.shape[0], ind[1] - arr.shape[1]))
        if zblock_l.shape[1] > 0:
            horz_list.insert(0, zblock_l)

        # block of zeros to the right of arr
        zblock_r = np.zeros((arr.shape[0], n - ind[1]))
        if zblock_r.shape[1] > 0:
            horz_list.append(zblock_r)

        block_list.append(horz_list)

    B = bmat(block_list)

    return B

def build_D11(weights, n):
    """
    Create the W' matrix as a block diagonal matrix with blocks strictly below the main diagonal.
    
    Parameters:
    weights: Either a nn.Sequential model or a list/tensor of weights
    n: The size of the square matrix (number of rows/columns)
    
    Returns:
    D11: torch.Tensor representing the W' matrix
    """
    # Initialize W' as a zero tensor
    D11 = torch.zeros(n, n)
    
    # Extract weights from the input
    if isinstance(weights, torch.nn.Sequential):
        # If weights is a Sequential model, extract weights from its layers
        weight_list = [layer.weight.data for layer in weights if hasattr(layer, 'weight')]
    elif isinstance(weights, (list, torch.Tensor)):
        # If weights is already a list or tensor, use it directly
        weight_list = weights if isinstance(weights, list) else weights.tolist()
    else:
        raise ValueError("Input must be either nn.Sequential, a list, or a tensor of weights")
    
    # Ensure we have the correct number of weights
    if len(weight_list) != n - 1:
        raise ValueError(f"Incorrect number of weights. Expected {n-1}, got {len(weight_list)}")
    
    # Fill the block diagonal elements of W' with the weights
    for i in range(1, n):
        D11[i, i-1] = weight_list[i-1]
    
    return D11

def schur(matrix, dim_A, dim_B, dim_C, dim_D):
    """
    Calcule le complément de Schur pour le bloc A d'une matrice donnée avec des blocs spécifiés.

    Arguments:
    matrix -- Matrice torch (2D) en entrée.
    dim_A -- Dimension de la matrice A (nombre de lignes et de colonnes pour un bloc carré).
    dim_B -- Dimension de la matrice B (nombre de lignes de A et de colonnes de B).
    dim_C -- Dimension de la matrice C (nombre de lignes de C et de colonnes de A).
    dim_D -- Dimension de la matrice D (nombre de lignes et de colonnes pour un bloc carré).

    Retourne:
    Le complément de Schur du bloc A.
    """

    # Extraction des blocs A, B, C, D
    A = matrix[:dim_A, :dim_A]
    B = matrix[:dim_A, dim_A:dim_A + dim_B]
    C = matrix[dim_A:dim_A + dim_C, :dim_A]
    D = matrix[dim_A:dim_A + dim_C, dim_A:dim_A + dim_D]

    # Calcul du complément de Schur S = A - B D^{-1} C
    D_inv = torch.inverse(D)
    S = A - torch.mm(torch.mm(B, D_inv), C)

    return S


def isSDP(L: torch.Tensor, tol=1e-9) -> bool:
    '''
    Check if a Tensor is Positive definite up to a fixed tolerance.

    '''
    isAllEigPos = torch.all(torch.real(eigvals(L)) > 0)
    isSymetric = torch.all(torch.abs(L - L.T) < tol)
    if not isAllEigPos:
        print("Not all eigenvalues are positive \n")
    if not isSymetric:
        print("Matrix is not symmetric \n")

    bSDP = bool(isSymetric and isAllEigPos)
    if bSDP:
        print("Matix is SDP \n")
    return bSDP


def getEigenvalues(L: torch.Tensor):
    '''
    Return the eigenvalues of a given Tensor

        params :
            - L pytorch Tensor
    '''
    return torch.linalg.eigvals(L)


def is_alpha_stable(A: torch.Tensor, alpha: torch.Tensor):
    """
    Check if all eigenvalues of A are negative and lower than - alpha

    """
    return torch.all(eigvals(A) < - alpha)


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

    return torch.cat((iIpA @ (I_nin - A), -2 * V @ iIpA), axis=1)     # type: ignore


def create_block_lower_triangular(block_sizes, device='cpu'):
    """
    Create a strictly block lower triangular matrix with non-zero blocks only on the first sub-diagonal.

    Args:
    block_sizes (list): A list of integers representing the sizes of each block.
    device (str): The device to create the tensor on ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The resulting block lower triangular matrix.
    """
    n_blocks = len(block_sizes)
    total_size = sum(block_sizes)

    # Create the full matrix of zeros
    matrix = torch.zeros(total_size, total_size, device=device)

    # Create and place the non-zero blocks on the first sub-diagonal
    start_row = 0
    start_col = 0
    for i in range(n_blocks - 1):
        row_size = block_sizes[i+1]
        col_size = block_sizes[i]
        block = torch.randn(row_size, col_size, device=device)

        start_row += block_sizes[i]
        matrix[start_row:start_row + row_size,
               start_col:start_col + col_size] = block
        start_col += col_size

    return matrix


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input) -> torch.Tensor:
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output) -> Union[torch.Tensor, None]:
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


def solveLipschitz(weights, beta=1, epsilon=1e-6, solver="MOSEK"):
    r'''
        This function solve the Linear Matrix Inequality for
        estimating an upper bound on the Lipschitz constant of
        a feedforward neural network without skip connections.
        https://arxiv.org/abs/2005.02929
        params :
            * weights : a list of neural network weights
            * beta : the maximum slope of activation functions
    '''
    n_in = weights[0].shape[1]
    n_hidden = [w.shape[0] for w in weights[:-1]]
    n_out = weights[-1].shape[0]

    Ts = [Variable((n_h, n_h), diag=True) for n_h in n_hidden]
    T = block_diag(Ts)
    Ft = bmat([[np.zeros(T.shape), beta * T],
              [beta * T, -2 * T]])  # type: ignore
    Ws = [weight.detach().numpy() for weight in weights[:-1]]
    W = block_diag(Ws)
    A = hstack([W, np.zeros((W.shape[0], n_hidden[-1]))])

    I_B = np.eye(sum(n_hidden))
    B = hstack([np.zeros((I_B.shape[0], n_in)), I_B])
    AB = vstack([A, B])
    LMI = AB.T @ Ft @ AB

    LMI_schur = block_diag([LMI, np.zeros((n_out, n_out))])

    # 2eme partie LMI
    lip = Variable()
    L = -lip * np.eye(n_in)  # type: ignore

    # Block schur last layer
    b11 = np.zeros((n_hidden[-1], n_hidden[-1]))
    b12 = weights[-1].T.detach().numpy()
    b21 = weights[-1].detach().numpy()
    b22 = -np.eye(n_out)
    bf = bmat([[b11, b12], [b21, b22]])

    dim_inter = sum(n_hidden[:-1])
    if dim_inter > 0:
        inter = np.zeros((dim_inter, dim_inter))
        part2 = block_diag([L, inter, bf])
    else:
        part2 = block_diag([L, bf])

    M = LMI_schur + part2

    nM = M.shape[0]
    nT = T.shape[0]
    constraints = [M << -np.eye(nM) * epsilon, T - (epsilon)
                   * np.eye(nT) >> 0, lip - epsilon >= 0]  # type: ignore
    objective = Minimize(lip)  # Find lowest lipschitz constant

    prob = Problem(objective, constraints=constraints)
    prob.solve(solver)
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print(" Lipschitz Constant upper bound (All layer versions): \n")
        print(np.sqrt(lip.value))

    else:
        raise ValueError("SDP problem is infeasible or unbounded")

    lip = torch.Tensor(np.array(np.sqrt(lip.value))).to(dtype=torch.float32)
    # Evaluate if it closed to the boundary of the LMI
    # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
    # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
    # If it is close to zero it is at the center
    # Ts = [torch.Tensor(tens.value.todense()).to(dtype=torch.float32) for tens in Ts]
    # T = torch.block_diag()
    T = torch.Tensor(T.value).to(dtype=torch.float32)
    M = torch.Tensor(M.value).to(dtype=torch.float32)
    return T, lip, M


# Thanks Lezcano again !
def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2 * n, 2 * n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)


def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)


class Logm(torch.autograd.Function):
    """
    Computes the matrix logarithm of a given sqaure matrix.
    """
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (torch.float32, torch.float64,
                           torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)


logm = Logm.apply
