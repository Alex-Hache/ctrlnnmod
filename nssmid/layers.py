import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
import geotorch


class CustomSoftplus(nn.Softplus):
    def __init__(self, beta: int = 1, threshold: int = 20, margin : float = 1e-2) -> None:
        super(CustomSoftplus, self).__init__(beta, threshold)
        self.margin = margin

    def forward(self, x):
        return F.softplus(x, self.beta, self.threshold) + self.margin

class BetaLayer(nn.Module):

    def __init__(self, n_inputs, n_states, n_hidden, actF = nn.Tanh(), func = 'softplus', tol = 0.01) -> None:

        '''
        This function initiates a "beta layer" whiches produce a matrix valued function 
        beta(x) where beta(x) is invertible and of size n_inputsx n_inputs : 
            * beta(x) = U sigma(x) V
            where U,V are orthogonal matrices and 
            sigma(x) a one-layer feedforward neural network  : pos_func(W_out \sigma(W_in x + b_in)+b_out)

            It is initialized to be Identity matrix
        '''
        super(BetaLayer, self).__init__()

        self.nu = n_inputs
        self.nx = n_states
        self.nh = n_hidden
        self.actF = actF
        self.U  = nn.Linear(self.nu, self.nx, bias = None)
        nn.init.eye_(self.U.weight)
        geotorch.orthogonal(self.U, "weight")

        self.V = nn.Linear(self.nx, self.nu, bias = None)
        nn.init.eye_(self.V.weight)
        geotorch.orthogonal(self.V, "weight")


        self.W_beta_in = nn.Linear(self.nx, self.nh)
        self.W_beta_out = nn.Linear(self.nh, self.nu)
        nn.init.zeros_(self.W_beta_out.weight)
        nn.init.zeros_(self.W_beta_out.bias)

        if func == 'softplus':
            self.pos_func = CustomSoftplus(beta=1, threshold=20, margin=tol)
        elif func == 'relu':
            self.pos_func = nn.ReLU()
        else:
            raise(NotImplementedError("Not implemented yet"))

    
    def forward(self, x):
        sig = self.W_beta_in(x)
        sig = self.actF(sig)
        sig = self.W_beta_out(sig)
        sig = self.pos_func(x)
        return self.U(sig.unsqueeze(-2) @ self.V.transpose(-2, -1))




class InertiaMatrix(nn.Module):

    def __init__(self, nq, nh, actF = nn.Tanh(), func = 'softplus', tol = 0.01) -> None:

        '''
        This function initiates a "beta layer" whiches produce a matrix valued function 
        beta(x) where beta(x) is invertible and of size n_inputsx n_inputs : 
            * beta(x) = U sigma(x) V
            where U,V are orthogonal matrices and 
            sigma(x) a one-layer feedforward neural network  : pos_func(W_out \sigma(W_in x + b_in)+b_out)

            It is initialized to be Identity matrix
        '''
        super(InertiaMatrix, self).__init__()

        self.nu = nq
        self.nh = nh
        self.actF = actF
        self.U  = nn.Linear(self.nu, self.nu, bias = None)
        nn.init.eye_(self.U.weight)
        geotorch.orthogonal(self.U, "weight")

        self.W_beta_in = nn.Linear(self.nu, self.nh)
        self.W_beta_out = nn.Linear(self.nh, self.nu)
        nn.init.zeros_(self.W_beta_out.weight)
        nn.init.zeros_(self.W_beta_out.bias)

        if func == 'softplus':
            self.pos_func = CustomSoftplus(beta=1, threshold=20, margin=tol)
        elif func == 'relu':
            self.pos_func = nn.ReLU()
        else:
            raise(NotImplementedError("Not implemented yet"))

    
    def forward(self, x):
        sig = self.W_beta_in(x)
        sig = self.actF(sig)
        sig = self.W_beta_out(sig)
        sig = self.pos_func(x)
        return self.U(sig.unsqueeze(-2) @ self.U.weight.transpose(-2, -1))



class CoriolisMatrix(nn.Module):

    def __init__(self, nq, nh, actF = nn.Tanh(), func = 'softplus', tol = 0.01) -> None:

        '''
        This function initiates a "beta layer" whiches produce a matrix valued function 
        beta(x) where beta(x) is invertible and of size n_inputsx n_inputs : 
            * beta(x) = U sigma(x) V
            where U,V are orthogonal matrices and 
            sigma(x) a one-layer feedforward neural network  : pos_func(W_out \sigma(W_in x + b_in)+b_out)

            It is initialized to be Identity matrix
        '''
        super(BetaLayer, self).__init__()

        self.nu = nq
        self.nh = nh
        self.actF = actF
        self.U  = nn.Linear(self.nu, self.nu, bias = None)
        nn.init.eye_(self.U.weight)
        geotorch.orthogonal(self.U, "weight")

        self.W_beta_in = nn.Linear(self.nu, self.nh)
        self.W_beta_out = nn.Linear(self.nh, self.nu)
        nn.init.zeros_(self.W_beta_out.weight)
        nn.init.zeros_(self.W_beta_out.bias)

        if func == 'softplus':
            self.pos_func = CustomSoftplus(beta=1, threshold=20, margin=tol)
        elif func == 'relu':
            self.pos_func = nn.ReLU()
        else:
            raise(NotImplementedError("Not implemented yet"))

    
    def forward(self, x):
        sig = self.W_beta_in(x)
        sig = self.actF(sig)
        sig = self.W_beta_out(sig)
        sig = self.pos_func(x)
        return self.U(sig.unsqueeze(-2) @ self.U.transpose(-2, -1))


