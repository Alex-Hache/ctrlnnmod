import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import geotorch_custom as geo
from ctrlnmod.linalg.utils import sqrtm
import numpy as np
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.atoms.affine.bmat import bmat
from ctrlnmod.utils import FrameCacheManager
from ctrlnmod.linalg import project_onto_stiefel
from typing import Dict, Union
from ctrlnmod.lmis.hinf import HInfCont
from .linear import SSLinear

class L2BoundedLinear(SSLinear):
    r"""
        Create a linear continuous-time state-space model with a prescribed L2 gain and alpha stability.
            \dot{x} = Ax + Bu + Gd
            y = Cx

        if there are exogenous signals we parameterize only the triplet (A,G,C)
        attributes
        ----------

            * nu : int
                input dimension
            * ny : int
                output dimension
            * nx : int
                state dimension
            * gamma : Tensor
                precribed L2 gain
            * alpha: float
                uuper bound on linear decay rate
            * Q : Tensor
                Positive definite matrix
            * S : Tensor
                skew-symmetric matrix
            * G : Tensor
                Output matrix
            * H : Tensor
                Semi-orthogonal matrix
    """
    def __init__(self, nu: int, ny: int, nx: int, gamma: float, nd: int = 0,
                 alpha: float = 0.0, scaleH=1.0, epsilon=0.0) -> None:
        super(L2BoundedLinear, self).__init__(nu, ny, nx, nd, alpha)
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.gamma = Tensor([gamma])
        self.alpha = Tensor([alpha])
        self.Ix = torch.eye(nx)
        self.scaleH = scaleH
        self.eps = epsilon
        self.nd = nd

        self.Q = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.P = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.S = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nx)))
        self.G = Parameter(nn.init.xavier_normal_(torch.empty(self.ny, self.nx)))
        

        # Register relevant manifolds
        geo.positive_definite(self, 'P')
        geo.positive_definite(self, 'Q')
        geo.skew_symmetric(self, 'S')
        

        if self.nd > 0:
            self.B = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nu)))
            self.H = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nd)))
        else:
            self.H = Parameter(nn.init.xavier_normal_(torch.empty(self.nx, self.nu)))
        geo.orthogonal(self, 'H')

        self._frame_cache = FrameCacheManager()
    
    def __repr__(self):
        return "Hinf_Linear_ss" + f"_alpha_{self.alpha}" + f"_gamma_{self.gamma}"

    def forward(self, u, x, d=None):
        if self.nd > 0 and d is not None:
            A, G, C = self._frame()
            B = self.B
            dx = x @ A.T + u @ B.T + d @ G.T
        else:
            A, B, C = self._frame()
            dx = x @ A.T + u @ B.T

        y = x @ C.T
        return dx, y

    def _frame(self) -> tuple[Tensor, Tensor, Tensor]:
        # Si la mise en cache est active et qu'un cache existe, retourner le cache
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        A = (-0.5 * (self.Q + self.G.T @ self.G + self.eps * self.Ix) + self.S) @ self.P - self.alpha * self.Ix
        B = self.gamma * sqrtm(self.Q) @ (self.scaleH * self.H)  # type: ignore
        C = self.G @ self.P

        # Stocker dans le cache si la mise en cache est active
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, B, C)
            
        return A, B, C

    def right_inverse_lmi(self, A, B, C, gamma: float, alpha):
        Q, P_torch, S, G, H, alph, _ = self.submersion_inv_lmi(A, B, C, float(gamma), alpha)
        self.P = P_torch
        self.Q = Q
        self.S = S
        self.G = Parameter(G)
        self.H = H
        self.alpha = alph

    def submersion_inv_lmi(self, A, B, C, gamma: float, alpha=0.0, epsilon=1e-8, solver="MOSEK", check=False):
        """
            Function from weights space to parameter space.

        """
        with torch.no_grad():
            A = A.detach().numpy()
            B = B.detach().numpy()
            C = C.detach().numpy()
            nx = A.shape[0]
            nu = B.shape[1]
            ny = C.shape[0]

            D = np.zeros((ny, nu))
            P = Variable((nx, nx), "P", PSD=True)
            gam = Variable()
            M = bmat(
                [
                    [A.T @ P + P @ A, P @ B, C.T],
                    [B.T @ P, -gam * np.eye(nu), D.T],  # type: ignore
                    [C, D, -gam * np.eye(ny)],  # type: ignore
                ]
            )
            constraints = [
                M << -epsilon * np.eye(nx + nu + ny),  # type: ignore
                P - (epsilon) * np.eye(nx) >> 0,  # type: ignore
                A.T @ P + P @ A + 2 * alpha * P << -(epsilon * np.eye(nx)),  # type: ignore
                gam - epsilon >= 0,  # type: ignore
            ]
            objective = Minimize(gam)  # Feasibility problem

            prob = Problem(objective, constraints=constraints)
            prob.solve(solver)
            if prob.status not in ["infeasible", "unbounded"]:
                gmma_lmi = gam.value
                if gmma_lmi > gamma and check:
                    raise ValueError(f"Infeasible problem with prescribed gamma : {gamma} min value = {gmma_lmi}")
                else:
                    if gmma_lmi > gamma:
                        print(
                            "Not in manifold with gamma = {} \n New gamma value assigned : g = {} -- alpha = {}".format(
                                gamma, gmma_lmi, alpha
                            )
                        )
                        self.gamma = float(gmma_lmi)  # Assign lowest gamma found if it's higher than the one prescribed
                    else:
                        print(f"Currrent gamma value : {gmma_lmi}")
                        self.gamma = gamma
            else:
                raise ValueError("SDP problem is infeasible or unbounded")



            # Now initialize
            P = Tensor(P.value)
            A = Tensor(A)
            B = Tensor(B)
            C = Tensor(C)
            # M_tilde = A.T @ P + P @ A + C.T @ C + (1/(gamma**2))* P @ B @ B.T @ P
            # A = (-0.5 * (self.Q + self.G.T @ self.G + self.eps * self.Ix) + self.S) @ self.P - self.alpha * self.Ix

            '''
                Version Riccati
            '''

            G = Tensor(C) @ torch.inverse(P)
            Q = Tensor(-M.value[:nx, :nx])  # type: ignore
            S = Tensor(A) @ torch.inverse(P) + 0.5 *(Q +G.T@G + self.eps * self.Ix) 
            H = Tensor(1 / self.gamma * (torch.inverse(sqrtm(Q)) @ B))
            H = Tensor(project_onto_stiefel(H))
            alph = Tensor([alpha])
        return Q, P, S, G, H, alph, gmma_lmi


    def right_inverse_(self, A, B, C, gamma: float, alpha):
        _, gamma, _ = HInfCont.solve(A, B, C, torch.zeros((C.shape[0], B.shape[1])), alpha = torch.Tensor([0.0]))
        Q, P_torch, S, G, H = self.submersion_inv_(A, B, C, float(gamma), alpha)
        self.P = P_torch
        self.Q = Q
        self.S = S
        self.G = Parameter(G)
        self.H = H
        self.alpha = alpha

    def init_weights_(self, A0, B0, C0, gamma: float, alpha, G0=None):
        if self.nd > 0  and G0 is not None:
            self.right_inverse_lmi(A0, G0, C0, gamma, alpha)
            self.B.data = B0
        else:
            self.right_inverse_lmi(A0, B0, C0, gamma, alpha)
    
    def submersion_inv_(self, A, B, C, gamma: float, epsilon=1e-8 ):
        """
            Function from weights space to parameter space.

        """


        # Now initialize
        A = Tensor(A)
        B = Tensor(B)
        C = Tensor(C)
        # M_tilde = A.T @ P + P @ A + C.T @ C + (1/(gamma**2))* P @ B @ B.T @ P
        # A = (-0.5 * (self.Q + self.G.T @ self.G + self.eps * self.Ix) + self.S) @ self.P - self.alpha * self.Ix

        '''
            Version Riccati
        '''

        U, S, Vt = torch.linalg.svd(1 / gamma * B)
        epsilon = 1e-6
        Q = U @ torch.block_diag(torch.diag(S)**2, epsilon*torch.eye(self.nx-self.nu)) @ U.T
        H = (1/gamma) * (torch.inverse(sqrtm(Q))) @ B
        # Consistency check
        P, infos = self.solve_riccati_torch(A, B, C, gamma)
        G = Tensor(C) @ torch.inverse(P)

        S = Tensor(A) @ torch.inverse(P) + 0.5 *(Q + G.T @ G)

        return Q, P, S, G, H
    

    def solve_riccati_torch(self, A: torch.Tensor, 
                        B: torch.Tensor, 
                        C: torch.Tensor, 
                        gamma: float,
                        tol: float = 1e-10) -> Dict[str, Union[torch.Tensor, bool, float]]:
        """
        Résout l'équation de Riccati pour la norme H∞ en utilisant la méthode hamiltonienne
        A^T P + PA + PBB^T P/γ² + C^T C = 0
        
        Args:
            A: Tensor de taille (nx, nx)
            B: Tensor de taille (nx, nu)
            C: Tensor de taille (ny, nx)
            gamma: Valeur scalaire > 0
            tol: Tolérance pour la stabilité
        
        Returns:
            Dict contenant:
                - 'P': Solution de l'équation de Riccati
                - 'success': Bool indiquant si la résolution a réussi
                - 'eigvals': Valeurs propres de la matrice hamiltonienne
                - 'residual_norm': Norme du résidu
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
            return {
                'success': False,
                'eigvals': eigvals,
                'error': str(e)
            }
    
    @classmethod
    def copy(cls, model):

        copy = __class__(
            model.nu,
            model.ny,
            model.nx,
            float(model.gamma),
            model.nd,
            float(model.alpha)
        )
        copy.load_state_dict(model.state_dict())
        return copy

    def clone(self):
        return L2BoundedLinear.copy(self)

    def check_(self):
        W = self._frame()
        try:
            _, _, _, _, _, _, gamma = self.submersion_inv(*W, float(self.gamma), check=True)
            return True, gamma
        except ValueError:
            return False, np.inf
