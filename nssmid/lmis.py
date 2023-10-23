import torch.nn as nn
from torch.nn.parameter import Parameter
import geotorch as geo
import cvxpy as cp
import numpy as np
import torch
from torch.nn import Module
from torch import Tensor

'''
    Continuous time LMI versions
'''
class LMI_Lyap(Module):
    '''
        A base class for all LMI based on Lyapunov stability for a continuous time linear system
        M = A^TP + PA 
        To create a new LMI make it child of this class and override init_lmi and forward methods.
        If more than 1 matrix for Lyapunov certificate add a method like symP_
    '''
    def __init__(self, A : torch.Tensor, epsilon : float = 1e-6) -> None:
        super(LMI_Lyap, self).__init__()

        self.P = self.P = Parameter(torch.empty_like(A))
        #self.P = Parameter(torch.empty_like(A))
        geo.positive_semidefinite(self, "P")
        M, P = self.init_lmi(A,epsilon = epsilon)
        self.P = Parameter(P.requires_grad_(True))
        self.A = A
        self.nm = M.shape[0]

    def init_lmi(self, A : torch.Tensor, epsilon = 1e-6, solver = "MOSEK"):
        A = A.detach().numpy()
        print(" Initializing Lyapunov LMI \n")
        nx = A.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)

        M = A.T@P + P@A


        constraints = [M << -np.eye(nx)*epsilon, P -(epsilon)*np.eye(nx)>> 0] 
        objective = cp.Minimize(0) # Feasibility problem

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        P = torch.Tensor(P.value).to(dtype=torch.float32)
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        return M, P

    def symP_(self):
        self.P = Parameter(0.5*(self.P + self.P.T))
    
    def forward(self):
        self.symP_()
        eig_val, _ = torch.linalg.eig(self.P)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")

        M = -(self.A.T@self.P + self.P@self.A)
        return M # M must be SDP 

class LMI_HInf(Module):
    def __init__(self, A :torch.Tensor, B :torch.Tensor, C :torch.Tensor, epsilon = 1e-6) -> None:
        super(LMI_HInf, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.P = nn.Linear(self.A.shape[0], self.A.shape[0], bias = False)

        P, gamma = self.solve_lmi(self.A,self.B, self.C, abs_tol=1e-5, solver ="MOSEK", epsilon= epsilon)
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
        M11 = torch.matmul(self.A.T, self.P.weight) + torch.matmul(self.P.weight, self.A) + 1/self.gamma*torch.matmul(self.C.T, self.C)
        M12 = torch.matmul(self.P.weight, self.B)
        
        M22 = -self.gamma*torch.eye(nu)

        M = torch.cat((torch.cat((M11,M12), 1), torch.cat((M12.T,M22), 1)), 0)
        return -M # Always have to return an positive semidefinite matrix

    def solve_lmi(self, A: torch.Tensor,B: torch.Tensor,C: torch.Tensor, abs_tol = 1e-6, solver = "MOSEK", espilon = 0.1):
        
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


        constraints = [M << -np.eye(nu+nx+ny)*espilon, P -(abs_tol)*np.eye(nx)>> 0, gamma - abs_tol >=0] 
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

    def symetrizeP_(self):
        self.P.weight = Parameter(0.5*(self.P.weight + self.P.weight.T))

class LMI_decay_rate(Module):
    def __init__(self, alpha : float, A :torch.Tensor, epsilon = 1e-6) -> None:
        super(LMI_decay_rate, self).__init__()

        self.alpha = alpha
        self.P = Parameter(torch.empty_like(A))
        geo.positive_semidefinite(self, "P")
        self.A = A
        M,P = self.solve_lmi(epsilon = epsilon)
        self.P = Parameter(P)

        self.nm = M.shape[0]

    def forward(self):
        eig_val, _ = torch.linalg.eig(self.P)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")

        M = -(self.A.T@self.P + self.P@self.A + 2*self.alpha*self.P)
        return M # M must be SDP

    def solve_lmi(self, epsilon = 1e-6, solver = "MOSEK"):
        
        A = self.A.detach().numpy()
        print(" Initializing Lyapunov LMI \n")
        alpha = self.alpha
        nx = A.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)

        M = A.T@P + P@A + 2*alpha*P


        constraints = [M << -np.eye(nx)*epsilon, P -(epsilon)*np.eye(nx)>> 0] 
        objective = cp.Minimize(0) # Feasibility problem

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        P = torch.Tensor(P.value).to(dtype=torch.float32)
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        return M, P

    def symP_(self):
        self.P = Parameter(0.5*(self.P + self.P.T))

class Lipschitz1Layer(Module):
    def __init__(self, upper_slope :float, W_out : Tensor, W_in : Tensor, init_margin : float = 1e-6) -> None:
        super(Lipschitz1Layer,self).__init__()
        self.alpha =  0.0
        self.beta = upper_slope
        self.w_out = W_out
        self.w_in = W_in

        M,T = self.solve_lmi(epsilon = init_margin)
        self.T = Parameter(T)
        self.nM = M.shape[0]

    def solve_lmi(self, epsilon = 1e-6, solver = "MOSEK"):

        print(" Initializing Lipschitz LMI \n")
        W_out = self.w_out.detach().numpy()
        W_in = self.w_in.detach().numpy()

        beta = self.beta

        n_in = self.w_in.shape[1]
        n_hid = self.w_in.shape[0]
        n_out = self.w_out.shape[0]

        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_hid

        T = cp.Variable((n_hid,n_hid), 'T', PSD=True)
        lip = cp.Variable() # square of the true lipschitz constant for solving the problem
        M11 = -lip*np.eye(n_in)
        M12 = beta*W_in.T@T
        M23 = W_out.T
        M = cp.bmat([[M11, M12, np.zeros((n_in,n_out))], 
                    [M12.T, -2*T, M23], [np.zeros((n_out,n_in)), M23.T, -np.eye(n_out, n_out)]])

        nM = n_in+n_hid+n_out
        constraints = [M << -np.eye(nM)*epsilon, T -(epsilon)*np.eye(nM)>> 0, lip-epsilon > 0] 
        objective = cp.Minimize(lip) # Feasibility problem

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print(" Lipschitz Constant upper bound : \n")
            
            print(lip.value)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        self.lip = torch.Tensor(np.sqrt(lip.value)).to(dtype=torch.float32)
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        T = torch.Tensor(T.value).to(dtype=torch.float32)
        return -M, T

    def forward(self):
        eig_val, _ = torch.linalg.eig(self.T)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("T is not SDP")

        M11 = self.lip*torch.eye(self.n_in)
        M12 = self.beta*self.w_out.T@self.T
        M13 = torch.zeros((self.n_in, self.n_out))
        M23 = self.w_out.T

        M = torch.cat((torch.cat((M11,M12,M13), 1), 
                       torch.cat((M12.T,-2*self.T, M23), 1),
                       torch.cat((M13.T, M23.T, -torch.eye(self.n_out)))), 0)
        return -M # M must be SDP

    def symT(self):
        self.T = Parameter(0.5*(self.T + self.T.T))


'''
    Discrete time LMI versions
'''

class LMI_HInf_Dist_discrete(Module):
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
