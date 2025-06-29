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
from typing import Union, List
from scipy.linalg import polar
from torch import Tensor




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

def get_lyap_exp(A) -> float:
    r"""
    Compute the Lyapunov exponent of a matrix.

    The Lyapunov exponent is defined as the maximum real part of the eigenvalues
    of a matrix :math:`A`. It characterizes the asymptotic stability of the
    linear system :math:`\dot{x} = A x`.

    .. math::
        \lambda = \max \Re(\lambda_i(A))

    where :math:`\lambda_i(A)` are the eigenvalues of :math:`A`.

    Args:
        A (torch.Tensor): A square matrix of shape (n, n).

    Returns:
        float: The Lyapunov exponent of the matrix.
    """
    return float(-torch.max(torch.real(torch.linalg.eigvals(A))))

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


def schur(matrix, dim_A, dim_B, dim_C):
    r"""
    Compute the Schur complement of the block D in a 2x2 partitioned matrix.

    The input matrix is assumed to be partitioned as:

    .. math::
        \begin{bmatrix}
            A & B \\
            C & D
        \end{bmatrix}

    where:
    - :math:`A` has shape (dim_A, dim_A)
    - :math:`B` has shape (dim_A, dim_B)
    - :math:`C` has shape (dim_C, dim_A)
    - :math:`D` has shape (dim_C, dim_B)

    The Schur complement is computed as:

    .. math::
        S = A - B D^{-1} C

    Args:
        matrix (torch.Tensor): A 2D tensor representing the full matrix.
        dim_A (int): Number of rows and columns of the top-left block A.
        dim_B (int): Number of columns of blocks B and D.
        dim_C (int): Number of rows of blocks C and D.

    Returns:
        torch.Tensor: The Schur complement of the block D.
    """

    # Shape verification
    expected_rows = dim_A + dim_C
    expected_cols = dim_A + dim_B
    if matrix.shape[0] < expected_rows or matrix.shape[1] < expected_cols:
        raise ValueError("Les dimensions fournies sont incompatibles avec la taille de la matrice.")

    # Blocks extraction
    A = matrix[:dim_A, :dim_A]
    B = matrix[:dim_A, dim_A:dim_A + dim_B]
    C = matrix[dim_A:dim_A + dim_C, :dim_A]
    D = matrix[dim_A:dim_A + dim_C, dim_A:dim_A + dim_B]

    # D must be square and inversible
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    
    try:
        D_inv = torch.inverse(D)
    except RuntimeError as e:
        raise ValueError("D must be inversible") from e

    # Schur : S = A - B D^{-1} C
    S = A - torch.matmul(torch.matmul(B, D_inv), C)

    return S



def is_positive_definite(L: torch.Tensor, tol=1e-3) -> bool:
    '''
    Check if a Tensor is Positive definite up to a fixed tolerance.

    Returns:
        bool: True if the matrix is positive definite with a maximum deviation from symmetry up to tol, False otherwise.
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
    Return the eigenvalues of a given Tensor L

        Args:
            torch.Tensor: the eigenvalues vector of L
    '''
    return torch.linalg.eigvals(L)


def is_alpha_stable(A: torch.Tensor, alpha: torch.Tensor):
    """
    Check if all eigenvalues of A are negative and lower than - alpha

    """
    return torch.all(eigvals(A) < - alpha)


def cayley(W):
    r"""
        Perform Cayley transform of rectangular matrix from 
        https://github.com/locuslab/orthogonal-convolutions

    """
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


import torch
from torch import Tensor
from typing import List, Literal

def fill_strictly_block_triangular(A: Tensor, blocks: List[Tensor], type: Literal['lower', 'upper'] = 'lower') -> Tensor:
    r"""
    Fill the matrix A on the first sub- or super-diagonal with the list of tensors in `blocks`.

    For example if A is of the form:
    .. math::
        A = \begin{bmatrix}
            A_{11} & A_{12} & A_{13} \\
            A_{21} & A_{22} & A_{23} \\
            A_{31} & A_{32} & A_{33} \\
        \end{bmatrix}

    and blocks contains [C, D], then it returns:
    .. math::
        A = \begin{bmatrix}
            A_{11} & A_{12} & A_{13} \\
            C & A_{22} & A_{23} \\
            A_{31} & D & A_{33} \\
        \end{bmatrix}  if type == 'lower'

    Args:
        A (Tensor): full 2D tensor to be modified in-place (assumed square block structure)
        blocks (List[Tensor]): list of blocks to insert along the first diagonal below or above the main
        type (str): 'lower' or 'upper' to determine which diagonal to fill

    Returns:
        Tensor: the modified matrix A
    """
    if type not in ['lower', 'upper']:
        raise ValueError("type must be either 'lower' or 'upper'")

    block_sizes = [block.shape for block in blocks]
    row_sizes, col_sizes = zip(*block_sizes)

    n_blocks = len(blocks) + 1
    assert A.shape[0] == sum(row_sizes) + col_sizes[0], "Matrix A row size mismatch"
    assert A.shape[1] == sum(col_sizes) + row_sizes[-1], "Matrix A column size mismatch"

    # Compute block start indices
    row_offsets = [0] + list(torch.cumsum(torch.tensor(row_sizes), dim=0).tolist())
    col_offsets = [0] + list(torch.cumsum(torch.tensor(col_sizes), dim=0).tolist())

    # Fill appropriate blocks
    for i, block in enumerate(blocks):
        if type == 'lower':
            row_idx = row_offsets[i + 1]
            col_idx = col_offsets[i]
        else:  # upper
            row_idx = row_offsets[i]
            col_idx = col_offsets[i + 1]

        A[row_idx:row_idx + block.shape[0], col_idx:col_idx + block.shape[1]] = block

    return A



def create_block_lower_triangular(block_sizes, device='cpu'):
    r"""
    Create a strictly block lower triangular matrix with non-zero blocks on the first sub-diagonal.

    Given a list of block sizes, this function generates a block lower triangular
    matrix where only the first sub-diagonal blocks are filled with random values
    and all other entries are zero.

    For example, for block sizes [2, 3, 4], the matrix has the structure:

    .. math::
        \begin{bmatrix}
            0 & 0 & 0 \\
            B_{21} & 0 & 0 \\
            0 & B_{32} & 0
        \end{bmatrix}

    where :math:`B_{21}` and :math:`B_{32}` are random matrices of appropriate dimensions.

    Args:
        block_sizes (list of int): Sizes of each block; the total matrix will have shape
            (sum(block_sizes), sum(block_sizes)).
        device (str): Device on which to create the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A block lower triangular matrix of shape (N, N), where N is the sum of block sizes.
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
        m = input.detach().cpu().numpy().astype(np.float64)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output) -> Union[torch.Tensor, None]:
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float64)
            gm = grad_output.data.cpu().numpy().astype(np.float64)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


# Thanks Mario Lezcano again !
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


def project_onto_stiefel(A: torch.Tensor):
    r"""
    Project a matrix onto the Stiefel manifold.

    The Stiefel manifold :math:`\mathrm{St}(n, p)` is the set of all :math:`n \times p`
    matrices with orthonormal columns. This function projects the input matrix
    :math:`A \in \mathbb{R}^{n \times p}` onto the Stiefel manifold using polar decomposition:

    .. math::
        A = U H, \quad \text{with } U \in \mathrm{St}(n, p)

    Args:
        A (torch.Tensor): A 2D tensor of shape (n, p) representing the matrix to be projected.

    Returns:
        numpy.ndarray: A matrix of shape (n, p) with orthonormal columns, lying on the Stiefel manifold.
    """
    A = A.detach().numpy()
    # Perform polar decomposition
    U, H = polar(A)
    
    # Verify orthogonality (for debugging)
    UTU = U.T @ U
    deviation = np.max(np.abs(UTU - np.eye(U.shape[1])))
    if deviation > 1e-10:
        print(f"Warning: Maximum deviation from orthonormality: {deviation}")
    
    return U


def solve_riccati_torch(A: torch.Tensor, 
                        B: torch.Tensor, 
                        C: torch.Tensor, 
                        gamma: float,
                        tol: float = 1e-10) -> tuple[Tensor, dict]:
        r"""
        Solve the continuous-time H-infinity Riccati equation using the Hamiltonian method.

        This function computes the solution :math:`P` to the Riccati equation:

        .. math::
            A^T P + P A + \frac{1}{\gamma^2} P B B^T P + C^T C = 0

        The solution is based on the spectral decomposition of the associated Hamiltonian matrix.

        Args:
            A (torch.Tensor): System matrix of shape (n_x, n_x).
            B (torch.Tensor): Input matrix of shape (n_x, n_u).
            C (torch.Tensor): Output matrix of shape (n_y, n_x).
            gamma (float): Positive scalar defining the H-infinity bound.
            tol (float, optional): Numerical tolerance for stability checks. Default is 1e-10.

        Returns:
            tuple:
                - P (torch.Tensor or None): The symmetric solution matrix of shape (n_x, n_x), or None if the computation fails.
                - info (dict): Dictionary containing:
                    - 'success' (bool): Whether the Riccati solution was successfully computed.
                    - 'eigvals' (torch.Tensor): Eigenvalues of the Hamiltonian matrix.
                    - 'residual_norm' (float, optional): Norm of the residual in the Riccati equation.
                    - 'is_positive' (bool, optional): Whether P is positive definite.
                    - 'cond_X' (float, optional): Condition number of the matrix X in the stable eigenspace.
                    - 'error' (str, optional): Error message if the computation failed.

        References:
            [1] https://web.stanford.edu/~boyd/papers/bisection_hinfty.html
            
        """


        nx = A.shape[0]
        gamma_sq_inv = 1/(gamma**2)
        
        # Construction de la matrice hamiltonienne
        H_11 = A
        H_12 = gamma_sq_inv * (B @ B.T)
        H_21 = -(C.T @ C)
        H_22 = -A.T
        
        H = torch.block_diag(H_11, H_22)
        H[:nx, nx:] = H_12
        H[nx:, :nx] = H_21
        
        # Calcul des valeurs et vecteurs propres
        # Note: torch.linalg.eig retourne les valeurs complexes même pour matrices réelles
        eigvals, eigvecs = torch.linalg.eig(H)
        
        # Conversion en valeurs réelles pour le tri
        real_parts = eigvals.real
        
        # Tri des valeurs propres par partie réelle
        sorted_indices = torch.argsort(real_parts)
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        
        # Vérification du nombre de valeurs propres stables
        n_stable = torch.sum(real_parts[sorted_indices] < -tol).item()
        
        if n_stable != nx:
            # Ajuster le seuil si nécessaire
            alternative_tol = abs(real_parts[nx-1].item()) * 10
            print(f"Adjusting tolerance from {tol} to {alternative_tol}")
            stable_indices = real_parts < alternative_tol
        else:
            stable_indices = real_parts < -tol
        
        # Extraire les vecteurs propres stables
        stable_eigvecs = eigvecs[:, stable_indices]
        
        if stable_eigvecs.shape[1] != nx:
            return {
                'success': False,
                'eigvals': eigvals,
                'error': f"Got {stable_eigvecs.shape[1]} stable eigenvectors, expected {nx}"
            }
        
        # Extraction de X et Y
        X = stable_eigvecs[:nx, :]
        Y = stable_eigvecs[nx:, :]
        
        # Vérifier le conditionnement de X
        try:
            cond_X = torch.linalg.cond(X).item()
        except:
            cond_X = float('inf')
        
        if cond_X > 1e12:  # seuil arbitraire
            print(f"Warning: X is poorly conditioned (cond = {cond_X:.2e})")
        
        try:
            # Utiliser la pseudo-inverse si X est mal conditionné
            if cond_X > 1e12:
                # Calcul manuel de la pseudo-inverse pour plus de contrôle
                U, S, Vh = torch.linalg.svd(X)
                S_pinv = torch.where(S > tol * S[0], 1/S, torch.zeros_like(S))
                X_pinv = (Vh.T.conj() * S_pinv.unsqueeze(0)) @ U.T.conj()
                P = Y @ X_pinv
            else:
                P = Y @ torch.linalg.inv(X)
                
            # Prendre la partie réelle et symétriser
            P = P.real
            P = (P + P.T)/2
            
            # Vérifier que P est définie positive
            try:
                L = torch.linalg.cholesky(P)
                is_positive = True
            except:
                is_positive = False
                print("Warning: P is not positive definite")
            
            # Calculer le résidu
            riccati_residual = (A.T @ P + P @ A + 
                            gamma_sq_inv * P @ B @ B.T @ P +
                            C.T @ C)
            residual_norm = torch.norm(riccati_residual).item()
            
            return P, {
                'success': True,
                'eigvals': eigvals,
                'residual_norm': residual_norm,
                'is_positive': is_positive,
                'cond_X': cond_X
            }
            
        except Exception as e:
            return None,{
                'success': False,
                'eigvals': eigvals,
                'error': str(e)
            }
        

def check_observability(A: torch.Tensor, C: torch.Tensor, tol: float = 1e-10) -> bool:
    """
    Check the observability of a system defined by matrices A and C.
    
    Args:
        A : torch.Tensor
            State transition matrix of shape (n, n).
        C : torch.Tensor
            Output matrix of shape (m, n).
        tol : float
            Tolerance for numerical stability.
        
    Returns:
        bool
            True if the system is observable, False otherwise.
    """
    n = A.shape[0]
    O = C.clone()
    
    for i in range(1, n):
        O = torch.cat((O, C @ torch.matrix_power(A, i)), dim=0)
    
    rank_O = torch.linalg.matrix_rank(O)
    
    return rank_O == n


def check_controllability(A: torch.Tensor, B: torch.Tensor, tol: float = 1e-10) -> bool:
    """
    Check the controllability of a system defined by matrices A and B.

    Args:
        A : torch.Tensor
            State transition matrix of shape (n, n).
        B : torch.Tensor
            Input matrix of shape (n, m).
        tol : float
            Tolerance for numerical stability.

    Returns:    
        bool
            True if the system is controllable, False otherwise.
    """
    n = A.shape[0]
    C = B.clone()
    
    for i in range(1, n):
        C = torch.cat((C, B @ torch.matrix_power(A, i)), dim=1)
    
    rank_C = torch.linalg.matrix_rank(C)
    
    return rank_C == n  