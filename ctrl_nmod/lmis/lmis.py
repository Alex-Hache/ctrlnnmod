import torch.nn as nn
from torch.nn.parameter import Parameter
import geotorch as geo
import cvxpy as cp
import numpy as np
import torch


class LMI_HInf_Dist_discrete(nn.Module):
    def __init__(self, A,B,C,gamma) -> None:
        super(LMI_HInf_Dist_discrete, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.P = nn.Linear(A.shape[0],A.shape[0])
        geo.PSD(self.P) # Positive semi definite
        self.gamma = gamma

    def forward(self):
        '''
         M = [A^TPA-P  A^T*P*B C^T
                * B^T*P*B -gammaI 0(nu,ny)
                *  *  -\gamma I_{nu}] <0
        ** returns : -M 
        
        '''
        M11 = torch.matmul(torch.matmul(self.A.T, self.P), self.A) - self.P
        M12 = torch.matmul(torch.matmul(self.A.T, self.P), self.B)
        M13 = self.C.T
        M22 = torch.matmul(torch.matmul(self.B.T, self.P), self.B) - self.gamma*torch.eye(nu)
        nu = self.B.shape[1]
        ny = self.C.shape[0]
        M23 = torch.zeros((nu,ny))
        M33 = self.gamma*torch.eye(ny)

        M = torch.Tensor([[M11, M12, M13],[M12.T, M22, M23],[M13.T, M23.T, M33]])
        return -M

class LMI_HInf(torch.nn.Module):
    def __init__(self, A :torch.Tensor, B :torch.Tensor, C :torch.Tensor, init_margin = 1e-6) -> None:
        super(LMI_HInf, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.P = nn.Linear(self.A.shape[0], self.A.shape[0], bias = False)

        P, gamma = self.init_lmi(self.A,self.B, self.C, abs_tol=1e-5, solver ="MOSEK", init_margin= init_margin)
        self.P.weight = Parameter(P)
        #geo.positive_definite(self.P, "weight")

        self.gamma = Parameter(gamma)
        print(f'Gamma prescrit = {float(self.gamma):.4e}')


    def forward(self):
        '''
         M = [A^TPA-P  A^T*P*B C^T
                * B^T*P*B -gammaI 0(nu,ny)
                *  *  -\gamma I_{nu}] <0
        ** returns : -M 
        
        '''
        self.symetrizeP_()
        eig_val, _ = torch.linalg.eig(self.P.weight)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")
        nu = self.B.shape[1]
        ny = self.C.shape[0]
        M11 = torch.matmul(self.A.T, self.P.weight) + torch.matmul(self.P.weight, self.A) + 1/self.gamma*torch.matmul(self.C.T, self.C)
        M12 = torch.matmul(self.P.weight, self.B)
        
        M22 = -self.gamma*torch.eye(nu)

        M = torch.cat((torch.cat((M11,M12), 1), torch.cat((M12.T,M22), 1)), 0)
        return -M # Always have to return an positive semidefinite matrix

    def init_lmi(self, A: torch.Tensor,B: torch.Tensor,C: torch.Tensor, abs_tol = 1e-6, solver = "MOSEK", init_margin = 0.1):
        
        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()
        print(" Initializing HInf LMI \n")

        nu = B.shape[1]
        nx = A.shape[0]
        ny = C.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)
        gamma = cp.Variable()
        D = np.zeros((ny,nu))


        M = cp.bmat([[A.T@P + P@A , P@B, C.T], [B.T@P, -gamma *np.eye(nu), D.T], [ C, D, -gamma*np.eye(ny)]])


        constraints = [M << -np.eye(nu+nx+ny)*init_margin, P -(abs_tol)*np.eye(nx)>> 0, gamma - abs_tol >=0] 
        objective = cp.Minimize(gamma) # Find the L2 gain of the BLA

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print("Gamma = {} \n".format(prob.value))
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return torch.Tensor(P.value).to(dtype=torch.float32), torch.Tensor(gamma.value).to(dtype=torch.float32)

    def solve_lmi(self, A: torch.Tensor,B: torch.Tensor,C: torch.Tensor, abs_tol = 1e-6, solver = "MOSEK"):
        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()

        print(" Solving HInf LMI \n")

        nu = B.shape[1]
        nx = A.shape[0]
        ny = C.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)
        gamma = cp.Variable()
        D = np.zeros((ny,nu))


        M = cp.bmat([[A.T@P + P@A , P@B, C.T], [B.T@P, -gamma*np.eye(nu), D.T], [ C, D, -gamma*np.eye(ny)]])


        constraints = [M << -np.eye(nu+nx+ny)*abs_tol, P -(abs_tol)*np.eye(nx)>> 0, gamma - (abs_tol)>=0 ] 
        objective = cp.Minimize(gamma) # Find the L2 gain of the BLA

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print("Gamma = {} \n".format(prob.value))
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        
        # Evaluate if it closed to the boundary of the LMI
        #X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return torch.Tensor(P.value).to(dtype=torch.float32), torch.Tensor(gamma.value).to(dtype=torch.float32)

    def symetrizeP_(self):
        self.P.weight = Parameter(0.5*(self.P.weight + self.P.weight.T))

