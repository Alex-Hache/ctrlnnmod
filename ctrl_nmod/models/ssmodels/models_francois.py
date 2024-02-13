import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
from ctrl_nmod.linalg.utils import isSDP
from torch.nn import Module
from torch.linalg import cholesky
import os
import time
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from ctrl_nmod.losses.losses import Mixed_MSELOSS
from torch.optim import lr_scheduler
from torch.linalg import eigvals

def standardize_(self, a = 0, b = 1):
    # Maximum
    self.u_max, _ = torch.max(self.u,dim=0)
    self.y_max, _ = torch.max(self.y, dim = 0)
    self.x_max, _ = torch.max(self.x, dim = 0)

    # Minimum
    self.u_min, _ = torch.min(self.u,dim=0)
    self.y_min, _ = torch.min(self.y, dim = 0)
    self.x_min, _ = torch.min(self.x, dim = 0)

    if not any(self.u_max==0.):
        self.u = a + (b-a)*(self.u - self.u_min)/(self.u_max - self.u_min).requires_grad_()
    if not any(self.y_max==0.):
        self.y = a + (b-a)*(self.y - self.y_min)/(self.y_max - self.y_min).requires_grad_()
    if not any(self.x_max==0.):
        if self.x.requires_grad:
            self.x = a + (b-a)* (self.x - self.x_min)/(self.x_max - self.x_min).requires_grad_()
        else:
            self.x = a + (b-a)*(self.x - self.x_min)/(self.x_max - self.x_min)


class Simulator(nn.Module):
    def __init__(self, ss_model, ts, nb: int) -> None:
        super(Simulator, self).__init__()

        self.ss_model = ss_model
        self.ts = ts
        self.nx = self.ss_model.nx
        self.str_savepath = ss_model.str_savepath
        self.nb = int(nb)

    def __str__(self) -> str:
        model = str(self.ss_model)
        return " Model :" + model

    def clone(self):
        copy_ss = self.ss_model.clone()  # State-space model module must have a clone function
        copy = type(self)(copy_ss, self.ts, self.nb)
        copy.load_state_dict(self.state_dict())
        return copy

    def save(self) -> None:
        torch.save(self, self.str_savepath)

    @staticmethod
    def load(path):
        return torch.load(path, weights_only=False)

    def get_obs_size(self):
        return self.nb  # Taille des u pour estimer l'état

    def simulate(self, u, x0=None):
        if len(u.shape) == 3 and x0 is None:
            n_batch = u.shape[0]
            x0_sim = torch.zeros(n_batch, self.nx)
            u_sim = u
        elif len(u.shape) == 2:

            if x0 is None:
                x0 = torch.zeros(self.nx)
                x0_sim = x0[None, :]
                u_sim = u[None, :, :]
            else:
                x0_sim = x0[None, :]
        else:
            x0_sim = x0
            u_sim = u
        start_time = time.time()
        x_sim, y_sim = self(u_sim, x0_sim)  # Appel à forward
        x_sim.squeeze_(0)
        y_sim.squeeze_(0)
        sim_time = time.time() - start_time
        print(f"\n Sim time: {sim_time:.2f}")
        return x_sim, y_sim

    def predict(self, dataset):
        '''
            Must predict y[t+1] :
            from x0 = 0 we simulate the model for nb+1 steps to reach
            x[k+2], y[k+1]

            Assuming all sequences in dataset here are correctly sliced
            they're all the same size and their size is at least nb
        '''
        # If not all exp are the same size
        # seq_len = torch.min(torch.Tensor([exp.shape[0] for exp in dataset.u]))

        # seq_len = dataset.u[0].shape[0]
        seq_len = self.nb+1
        u_sim = torch.zeros(len(dataset.u), seq_len, dataset.u[0].shape[1])  # Nb x seq_len x nu
        x0_sim = torch.zeros(len(dataset.u), self.nx)
        for k, (u, y) in enumerate(dataset.iter_exp()):
            u_sim[k, :self.nb, :] = torch.Tensor(u[:self.nb, :])
        y_pred = []
        _, y_pred = self.simulate(u_sim, x0_sim)
        return list(y_pred[:, -1, :])

    def forecast(self, dataset):
        '''
            Must return the whole simulated sequence
        '''
        with torch.no_grad():
            y_sim = []
            x_sim = []
            for u, _ in dataset.iter_exp():
                u_sim = torch.Tensor(u)
                x_sim_exp, y_sim_exp = self.simulate(u_sim[None, :, :])
                x_sim.append(x_sim_exp)
                y_sim.append(y_sim_exp)
        return x_sim, y_sim

    def evaluate(self, test_set, sim_res=False):
        y_true = test_set.y
        with torch.no_grad():
            val_mse = torch.Tensor([0.0])
            _, y_sim = self.forecast(test_set)
            for k, exp in enumerate(y_true):
                y_true_torch = torch.Tensor(exp)
                val_mse += torch.mean((y_true_torch-y_sim[k])**2)
        if sim_res:
            return val_mse/(k+1), y_sim
        else:
            return val_mse/(k+1)

    def fit(self, train_dataset, test_dataset, epochs, criterion=Mixed_MSELOSS(alpha=0), seq_len=30,
            lr=1e-3, train_dir=os.getcwd(), batch_size=256, test_freq=500,
            tol_change=0.01, patience=10):

        nparams = np.sum([p.numel() for p in self.parameters() if p.requires_grad])

        if nparams >= 1000000:
            print(f"name: {self.__repr__()}, num_params: {1e-6*nparams:.1f}M")
        else:
            print(f"name: {self.__repr__()}, num_params: {1e-3*nparams:.1f}K")

        # Prepare data
        u_train, y_train, _ = train_dataset.get_dense_data()
        u_train = torch.Tensor(u_train)
        y_train = torch.Tensor(y_train)
        x_train = torch.zeros(u_train.shape[0:2] + (self.nx,)).requires_grad_(True)
        '''
        # Batch extraction funtions
        def get_batch(batch_size, seq_len):

            rng = np.random.default_rng()

            # Select batch indexes
            num_train_samples = u_train.shape[1]
            batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
            batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len)  # batch samples indices

            # Extract batch data
            batch_x0_hidden = x_train[:, batch_start, :]
            batch_x_hidden = x_train[:, batch_idx]
            batch_u = u_train[:, batch_idx]
            batch_y = y_train[:, batch_idx]

            return batch_x0_hidden, batch_u, batch_y, batch_x_hidden
        '''

        n_samples = u_train.shape[1]
        n_exp = u_train.shape[0]

        def get_batch(batch_size, seq_len):

            # Generate random indices for the first two dimensions
            exp_indices = np.random.choice(n_exp, size=batch_size, replace=True)
            sample_indices = np.random.choice(n_samples - seq_len + 1, size=batch_size, replace=True)

            batch_x0_hidden = x_train[exp_indices, sample_indices, :]
            # Advanced indexing shit thanks gpt

            batch_x_hidden = x_train[exp_indices[:, None], np.arange(seq_len)[None, :] + sample_indices[:, None], :]
            batch_u = u_train[exp_indices[:, None], np.arange(seq_len)[None, :] + sample_indices[:, None], :]
            batch_y = y_train[exp_indices[:, None], np.arange(seq_len)[None, :] + sample_indices[:, None], :]

            return batch_x0_hidden, batch_u, batch_y, batch_x_hidden
        # x_sim_train, _ = self.forecast(train_dataset)
        # x_train = torch.Tensor(x_sim_train[0].unsqueeze(0))
        # Setup optimizer
        lr = lr
        params_net = list(self.parameters())
        params_hidden = [x_train]
        optimizer = torch.optim.AdamW([
            {'params': params_net,    'lr': lr},
            {'params': params_hidden, 'lr': lr},
        ], lr=lr)
        # sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
        #                                       patience=1000, min_lr=1e-4, verbose=True)
        val_mse = self.evaluate(test_dataset)
        if torch.isnan(val_mse):
            val_mse = float(torch.inf)
        print("Initial val_MSE = {:.9f} \n".format(float(val_mse)))
        torch.save(self.state_dict(), f"{train_dir}/model.ckpt")

        vLoss = []
        vVal_mse = []
        '''
        E = self.ss_model.sys.E
        F = self.ss_model.sys.F
        A = torch.inverse(E) @ F
        vE = [E.detach().numpy()]
        vF = [F.detach().numpy()]
        vA = [A.detach().numpy()]
        A_true = np.array([[0.6, 1], [0, -0.25]])
        vDA = [np.linalg.norm(A.detach().numpy() - A_true)]
        vEigA = [torch.abs(eigvals(A)).detach().numpy()]
        '''
        start_time = time.time()
        # Training loop
        best_loss = val_mse
        # best_model = self.clone()
        no_decrease_counter = 0
        test_freq = test_freq
        tol_change = tol_change

        with alive_bar(epochs) as bar:
            epoch_loss = 0.0
            for itr in range(0, epochs):
                optimizer.zero_grad()

                # Simulate
                batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
                x_sim_torch_fit, y_sim_torch_fit = self(batch_u, batch_x0_hidden)

                # Compute fit loss
                loss = criterion(batch_y, y_sim_torch_fit, batch_x_hidden, x_sim_torch_fit)

                epoch_loss += loss.item()

                if itr % test_freq == 0 or itr == epochs-1:
                    # Statistics
                    epoch_loss = epoch_loss/test_freq
                    vLoss.append(epoch_loss)

                    val_mse = self.evaluate(test_dataset)
                    vVal_mse.append(float(val_mse))
                    '''
                    E = self.ss_model.sys.E
                    F = self.ss_model.sys.F
                    A = torch.inverse(E) @ F
                    vE.append(E.detach().numpy())
                    vF.append(F.detach().numpy())
                    vA.append(A.detach().numpy())
                    vEigA.append(torch.abs(eigvals(A)).detach().numpy())
                    vDA.append(np.linalg.norm(A.detach().numpy() - A_true))
                    '''
                    if (best_loss - val_mse)/best_loss > tol_change:
                        no_decrease_counter, best_loss = 0, float(val_mse.clone())
                        torch.save(self.state_dict(), f"{train_dir}/model.ckpt")  
                    else:
                        no_decrease_counter += 1

                    print(" Epoch loss = {:.7f} || Val_MSE = {:.7f} || Best loss = {:.7f} \n".format(float(epoch_loss),
                          float(val_mse), float(best_loss)))
                    epoch_loss = 0.0
                    if no_decrease_counter > patience/5 and hasattr(criterion, 'mu'):
                        criterion.update_mu_(0.1)
                        print(f"Updating barrier term weight : mu = {criterion.mu}")
                        no_decrease_counter = 0
                    if hasattr(criterion, 'mu') and criterion.mu < 1e-8:
                        break
                    if no_decrease_counter > patience:  # early stopping
                        break
                if (torch.isnan(loss)):
                    break
                # Optimize
                loss.backward()
                optimizer.step()
                # sched.step(val_mse)
                bar()

        train_time = time.time() - start_time

        print("Total dentification runtime : {} \n Best loss : {} \n".format(train_time, best_loss))

        model_state = torch.load(f"{train_dir}/model.ckpt")
        self.load_state_dict(model_state)
        # Final simulation perf on test data
        val_mse, y_sim_val = self.evaluate(test_dataset, sim_res=True)

        # Final simulation on train data
        train_mse, y_sim_train = self.evaluate(train_dataset, sim_res=True)
        print(f"Final MSE = {float(train_mse.detach()):.7f} || Val_MSE = {float(val_mse):.7f} \n")

        # Evaluate constraints
        self.check()
        strSaveName = os.path.join(os.getcwd(), 'results', 'sim_discrete_node_ren' + f'lr_{lr}' + f'_{epochs}epch')

        for k, y_exp in enumerate(y_train):
            fig = plt.figure()
            plt.plot(y_sim_train[k], label='model')
            plt.plot(y_exp, label='data')
            plt.legend()
            plt.show()

            fig.savefig(f"{strSaveName}_sim_exp_{k}.png")

        fig = plt.figure()

        plt.plot(range(len(vLoss)), np.log10(np.array(vLoss)), label='Loss')
        plt.plot(range(len(vVal_mse)), np.log10(np.array(vVal_mse)), label='Test loss') 

        # Ajouter une légende
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("MSE (log10)")
        plt.title(f"{str(self)}")
        plt.show()

        fig.savefig(f"{strSaveName}.png")

        dict_res = {'train_loss': vLoss,
                    'test_loss': vVal_mse,
                    'y_sim': [y_exp.squeeze(0).numpy() for y_exp in y_sim_val],
                    'y_sim_train': [y_exp.squeeze(0).numpy() for y_exp in y_sim_train],
                    'train_mse': train_mse.detach().numpy(),
                    'val_mse': val_mse.detach().numpy()}

        savemat(f"{strSaveName}" + " results.mat", dict_res)

        '''
        fig = plt.figure()
        plt.plot(range(len(vEigA)), vEigA, label='Eigenvalues of A')
        plt.plot(range(len(vDA)), vDA, label="Distance to A true")
        plt.show()'''
        return self, dict_res

    def extract_weights(self):
        weights = []
        biases = []
        for param_tensor in self.state_dict():
            tensor = self.state_dict()[param_tensor].detach().numpy()

            if 'weight' in param_tensor:
                weights.append(tensor)
            if 'bias' in param_tensor:
                biases.append(tensor)
        weights = np.array(weights, dtype=object)
        biases = np.array(biases, dtype=object)
        return weights, biases

    def write_flat_params(self, x):
        r""" Writes vector x to model parameters.."""
        index = 0
        theta = torch.Tensor(x)
        for name, p in self.named_parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()

    def flatten_params(self):
        views = []
        for i, p in enumerate(self.parameters()):
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)


class NNLinear(nn.Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim :int, output_dim: int, nx, strSavpath = os.getcwd()) -> None:
        super(NNLinear, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim

        self.A = nn.Linear(self.nx, self.nx, bias=False)
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.A.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator 
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True


class ImplicitNNLinear(nn.Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim: int, output_dim: int, nx, strSavpath=os.getcwd()) -> None:
        super(ImplicitNNLinear, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim

        self.E = Parameter(torch.Tensor(torch.randn(self.nx, self.nx)))
        self.F = nn.Linear(self.nx, self.nx, bias=False)
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        dx = (self.F(x) + self.B(u)) @ torch.inverse(self.E).T
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.F.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator 
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True


class ParamImplicitNNLinear(nn.Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim: int, output_dim: int, nx, strSavpath=os.getcwd()) -> None:
        super(ParamImplicitNNLinear, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim

        self.X = Parameter(torch.Tensor(torch.randn(2*self.nx, 2*self.nx)))
        self.Y = Parameter(torch.Tensor(torch.randn(self.nx, self.nx)))
        self.epsilon = 1e-4
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        nx = self.nx
        H = self.X @ self.X.T + self.epsilon * torch.eye(2*self.nx)
        H11 = H[:nx, :nx]
        P = H[nx:, nx:]
        F = H[nx:, :nx]
        E = 0.5*(H11 + P + self.Y - self.Y.T)
        dx = x @ (torch.inverse(E) @ F).T + u @ (torch.inverse(E) @ self.B.weight).T
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.F.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator 
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True


class ParamImplicitNNLinearAlph(nn.Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim: int, output_dim: int, nx, alpha: float = 1.0, strSavpath=os.getcwd()) -> None:
        super(ParamImplicitNNLinearAlph, self).__init__()

        self.nu = input_dim
        self.nx = nx
        self.ny = output_dim
        self.alpha = alpha

        self.X = Parameter(torch.Tensor(torch.randn(2*self.nx, 2*self.nx)))
        self.Y = Parameter(torch.Tensor(torch.randn(self.nx, self.nx)))
        self.epsilon = 1e-4
        self.B = nn.Linear(self.nu, self.nx, bias=False)
        self.C = nn.Linear(self.nx, self.ny, bias=False)
        # self.config = config
        self.str_savepath = strSavpath

    def forward(self, u, x):
        nx = self.nx
        H = self.X @ self.X.T + self.epsilon * torch.eye(2*self.nx)
        H11 = H[:nx, :nx]
        P = H[nx:, nx:]
        F = H[nx:, :nx]
        E = 0.5*(H11 + 1/(self.alpha**2)*P + self.Y - self.Y.T)
        dx = x @ (torch.inverse(E) @ F).T + u @ (torch.inverse(E) @ self.B.weight).T
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.F.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator 
        copy = type(self)(self.nu, self.ny, self.nx)
        copy.load_state_dict(self.state_dict())
        return copy

    def _check(self):
        return True


class Sim_discrete(Simulator):
    def __init__(self, ss_model, nb):
        super(Sim_discrete, self).__init__(ss_model=ss_model, ts=1, nb=nb)

    def forward(self, u_batch, x0_batch=torch.Tensor(1)):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        u_batch: Tensor. Size: (batch_size, seq_len, n_u)
            Input sequence for each subsequence in the minibatch

        x0_batch: Tensor. Size: (batch_size, n_x)
            initial state for each sequence in the minibatch
        x_batch: Tensor. Size: (batch_size, seq_len, n_x)
            state sequence for each subsequence in the minibatch
        Returns
        -------
        Tensor. Size: (batch_size, seq_len, n_x)
            Simulated state for all subsequences in the minibatch

        """

        X_sim_list = []
        Y_sim_list = []
        x_step = x0_batch
        for u_step in u_batch.split(1, dim=1):  # i in range(seq_len):

            u_step = u_step.squeeze(1)
            X_sim_list += [x_step]

            x_step, y_step = self.ss_model(u_step, x_step)
            Y_sim_list += [y_step]

        X_sim = torch.stack(X_sim_list, 1)
        Y_sim = torch.stack(Y_sim_list, 1)
        return X_sim, Y_sim

    def check(self):
        return self.ss_model._check()


class _ImplicitQSRNetwork(Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, S, Q, R,
                 gamma, device, bias=False, alpha=0.0, feedthrough=True) -> None:
        super().__init__()

        # Dimensions of Inputs, Outputs, States

        self.nx = nx        # no. internal-states
        self.ny = ny        # no. output
        self.nu = nu        # no. inputs
        self.nq = nq        # no. non-linear states
        self.s = np.max((nu, ny))
        self.epsilon = 0.0
        self.gamma = gamma
        self.device = device
        self.alpha = alpha
        self.feedthrough = feedthrough
        self.epsilon = epsilon

        # Initialization of the Weights:
        self.D12_tilde = Parameter(torch.randn(nq, nu, device=device))  # D12 = Lambda^{-1} DD12
        self.X3 = Parameter(torch.randn(self.s, self.s, device=device))
        self.Y3 = Parameter(torch.randn(self.s, self.s, device=device))
        self.Y1 = Parameter(torch.randn(nx, nx, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))
        BIAS = bias
        if (BIAS):
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)

        self.X = Parameter(torch.randn(2*nx+nq, 2*nx+nq, device=device))

        # Initialization of the last Parameters which are constrained:
        self.F = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.D12 = torch.zeros(nq, nu, device=device)
        self.D22 = torch.zeros(ny, nu, device=device)
        self.E = torch.zeros(nx, nx, device=device)
        self.Lambda = torch.zeros(nq, nq, device=device)

        self.R = R
        self.Q = Q
        self.S = S

        if self.feedthrough:
            try:
                self.Lq = cholesky(-Q)
            except torch.linalg.LinAlgError:
                self.Q = Q - self.epsilon * torch.eye(Q.shqpe[0])
                self.Lq = cholesky(-self.Q)

            self.Lr = cholesky(self.R - self.S @ torch.inverse(self.Q) @ self.S.T)

        self._frame()  # Update from parameters to weight space
        # Choosing the activation function:
        if (sigma == "tanh"):
            self.act = nn.Tanh()
        elif (sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif (sigma == "relu"):
            self.act = nn.ReLU()
        elif (sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def __str__(self):
        return f"ImplicitQSRNet_alpha_{self.alpha}_gamma_{self.gamma}"

    def _frame(self) -> None:
        M = F.linear(self.X3, self.X3) + (self.Y3 - self.Y3.T +
                                          (self.epsilon * torch.eye(self.s, device=self.device)))
        if self.feedthrough:  # Only update D22 if direct feedthrough
            M_tilde = F.linear(torch.eye(self.s, device=self.device) - M,
                               torch.inverse(torch.eye(self.s, device=self.device)+M).T)
            M_tilde = M_tilde[0:self.ny, 0:self.nu]

            self.D22 = torch.inverse(self.Q) @ self.S.T + torch.inverse(self.Lq) @ M_tilde @ self.Lr

        R_capital = self.R + self.S @ self.D22 + self.D22.T @ self.S.T + self.D22.T @ self.Q @ self.D22

        C2_tilde = (self.D22.T @ self.Q + self.S) @ self.C2
        D21_tilde = (self.D22.T @ self.Q) @ self.D21 - self.D12_tilde.T
        V_tilde = torch.cat([C2_tilde.T, D21_tilde.T, self.B2], 0)
        vec_C2_D21 = torch.cat([self.C2.T, self.D21.T, torch.zeros((self.nx, self.ny))], 0)

        Hs1 = self.X @ self.X.T
        Hs2 = self.epsilon*torch.eye(2*self.nx+self.nq, device=self.device)
        H_cvx = Hs1 + Hs2

        Hs3 = V_tilde @ torch.linalg.solve(R_capital, V_tilde.T)
        Hs4 = vec_C2_D21 @ self.Q @ vec_C2_D21.T

        H = H_cvx + Hs3 - Hs4

        # Partition of H in --> [H1 H2;H3 H4]
        h1, h2, h3 = torch.split(H, [self.nx, self.nq, self.nx], dim=0)  # you split the matrices in three big rows
        H11, H12, H13 = torch.split(h1, [self.nx, self.nq, self.nx], dim=1)  # you split each big row in 3 chunks
        H21, H22, H23 = torch.split(h2, [self.nx, self.nq, self.nx], dim=1)
        H31, H32, H33 = torch.split(h3, [self.nx, self.nq, self.nx], dim=1)

        self.F = H31
        self.B1 = H32
        self.P = H33
        self.C1 = -H21
        self.E = 0.5*(H11 + (1/self.alpha**2)*self.P + self.Y1 - self.Y1.T)
        self.Lambda = 0.5*torch.diag(torch.diag(H22))  # Faster for matrix but not available for general tensors
        # Lambda = 0.5*torch.diag_embed(torch.diagonal(H22))  # Equivalent who is the fastest ?

        L = -torch.tril(H22, -1)
        lambda_inv = torch.inverse(self.Lambda)
        self.D11 = lambda_inv @ L
        self.D12 = lambda_inv @ self.D12_tilde

    def forward(self, x, u):
        self._frame()
        # Then solve w_k
        # w = torch.zeros(x.shape[0], self.nq)
        # self.E = torch.eye(2)
        # self.F = torch.Tensor([[0.5, 1], [0, -0.25]])
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.E)
        dx = x @ (E_inv @ self.F).T + w @ (E_inv @ self.B1).T + u @ (E_inv @ self.B2).T
        if self.feedthrough:
            y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        else:
            y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x, u):
        nb = x.shape[0]  # batch size
        w = torch.zeros(nb, self.nq)
        v = torch.zeros(nb, self.nq)

        # Lambda v_k = C1 x_k + D11 w_k + D12 u_k

        for k in range(0, self.nq):
            v[:, k] = (1/self.Lambda[k, k]) * (x @ self.C1[k, :] + w.clone() @ self.D11[k, :]
                                               + u @ self.D12[k, :] + self.bv[k])  # 1 dimension no need fortranspose
            w[:, k] = self.act(v[:, k].clone())
        return w

    def _check(self) -> bool:
        self._frame()
        '''
            On construit 20 à partir des poids du réseau
        '''
        H11 = self.E + self.E.T - (1/self.alpha**2) * self.P

        H21 = -self.C1
        H12 = H21.T

        H31 = self.S @ self.C2
        H13 = H31.T

        H22 = 2*self.Lambda - self.Lambda @  self.D11 - self.D11.T @ self.Lambda

        H32 = self.S @ self.D21 - self.D12_tilde.T
        H23 = H32.T
        H33 = self.R + self.S @ self.D22 + (self.S @ self.D22).T

        H1 = torch.cat([H11, H12, H13], dim=1)
        H2 = torch.cat([H21, H22, H23], dim=1)
        H3 = torch.cat([H31, H32, H33], dim=1)

        H = torch.cat([H1, H2, H3], dim=0)

        # QSR dissipativity part
        J = torch.cat([self.F.T, self.B1.T, self.B2.T], dim=0)
        K = torch.cat([self.C2.T, self.D21.T, self.D22.T], dim=0)
        # diss = -J @ torch.inverse(self.P) @ J.T + K @ self.Q @ K.T  # Inverse of P is not symmetric
        diss = -J @ torch.linalg.solve(self.P, J.T) + K @ self.Q @ K.T  # A little bit more stable
        M = H + diss
        return isSDP(M, tol=1e-6)


class _System_general(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias=False, linear_output=False):
        """Used by the upper class NODE_REN. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -linear_output (bool, optional): choose if the output is linear,
            i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        # Dimensions of Inputs, Outputs, States

        self.nx = nx        # no. internal-states
        self.ny = ny        # no. output
        self.nu = nu        # no. inputs
        self.nq = nq        # no. non-linear states
        self.epsilon = epsilon
        self.device = device
        # Initialization of the Weights:
        self.D12_tilde = Parameter(torch.randn(nq, nu, device=device))  # D12 = Lambda^{-1} DD12
        self.Y1 = Parameter(torch.zeros(nx, nx, device=device))
        self.B2 = Parameter(torch.randn(nx, nu, device=device))
        self.C2 = Parameter(torch.randn(ny, nx, device=device))
        self.D21 = Parameter(torch.randn(ny, nq, device=device))
        BIAS = bias
        if (BIAS):
            self.bx = Parameter(torch.randn(nx, 1, device=device))
            self.bv = Parameter(torch.randn(nq, 1, device=device))
            self.by = Parameter(torch.randn(ny, 1, device=device))
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)

        self.X = Parameter(torch.eye(2*nx+nq, 2*nx+nq, device=device))

        # Initialization of the last Parameters which are constrained:
        self.F = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.D12 = torch.zeros(nq, nu, device=device)
        self.D22 = torch.zeros(ny, nu, device=device)
        self.E = torch.zeros(nx, nx, device=device)
        self.Lambda = torch.zeros(nq, nq, device=device)
        # Choosing the activation function:
        if (sigma == "tanh"):
            self.act = nn.Tanh()
        elif (sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif (sigma == "relu"):
            self.act = nn.ReLU()
        elif (sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def _frame(self):
        pass  # No need for updating weights

    def _check(self) -> bool:
        return True

    def forward(self, x, u):

        # Then solve w_k
        w = self._solve_w(x, u)
        E_inv = torch.inverse(self.E)
        dx = x @ (E_inv @ self.F).T + w @ (E_inv @ self.B1).T + u @ (E_inv @ self.B2).T
        if self.feedthrough:
            y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T
        else:
            y = x @ self.C2.T + w @ self.D21.T
        return dx, y

    def _solve_w(self, x, u):
        nb = x.shape[0]  # batch size
        w = torch.zeros(nb, self.nq)
        v = torch.zeros(nb, self.nq)

        # Lambda v_k = C1 x_k + D11 w_k + D12 u_k

        for k in range(0, self.nq):
            v[:, k] = (1/self.Lambda[k, k]) * (x @ self.C1[k, :].T + w @ self.D11[k, :].T 
                                               + u @ self.D12[k, :].T + self.bv[k])
            w[:, k] = self.act(v[:, k])
        return w


class NODE_REN(nn.Module):
    def __init__(self, nx=5, ny=5, nu=5, nq=5,
                 sigma="tanh", epsilon=1.0e-2, mode="c", gamma=1.,
                 device="cpu", bias=False, ni=1., rho=1., alpha=0.0,
                 linear_output=False, feedthrough=True, str_save=None):
        """Base class for Neural Ordinary Differential Equation Recurrent Equilbrium Networks (NODE_RENs).

        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function.
            It is possible to choose: 'tanh','sigmoid','relu','identity'.
            Defaults to "tanh".
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            Defaults to 1.0e-2.
            -mode (str, optional): Property to ensure.
            Possible options:
                -'c'= contractive model
                -'rl2'=L2 lipschitz-bounded,
                -'input_p'=input passive model,
                -'output_p'=output_passive model.
            -gamma (float, optional): If the model is L2 lipschitz bounded (i.e., mode == 'rl2'),
             gamma is the L2 Lipschitz constant. Defaults to 1.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -ni  (float, optional): If the model is input passive (i.e., mode == 'input_p') ,
            ni is the weight coefficient that characterizes the (input passive) supply rate function.
            -rho (float, optional): If the model is output passive (i.e., mode == 'output_p'),
            rho is the weight coefficient that characterizes the (output passive) supply rate function.
            -alpha (float, optional): Lower bound of the Contraction rate.
            If alpha is set to 0, the system continues to be contractive,
            but with a generic (small) rate. Defaults to 0.
            -linear_output (bool, optional): choose if the output is linear,
            i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        self.mode = mode.lower()
        self.sigma = sigma
        self.epsilon = epsilon
        self.gamma = gamma
        self.mode = mode
        self.device = device
        self.bias = bias
        self.ni = ni
        self.rho = rho
        self.alpha = alpha
        self.linear_output = linear_output
        self.feedthrough = feedthrough
        self.str_save = str_save

        if str_save is None:
            self.str_savepath = os.path.join(os.getcwd(), 'model' + self.mode + '.pkl')
        else:
            self.str_savepath = str_save

        if (self.mode == "general"):
            self.sys = _System_general(nx, ny, nu, nq,
                                       sigma, epsilon, device=device, bias=bias,
                                       linear_output=linear_output)
        else:  # QSR
            if self.mode == "rl2":
                Q = -(1./gamma)*torch.eye(ny, device=device)
                R = (gamma)*torch.eye(nu, device=device)
                S = torch.zeros(nu, ny, device=device)

            elif (self.mode == "input_p"):
                assert nu == ny, "Input and Output dimensions must be equal"
                Q = torch.zeros(ny, device=device)
                R = -2*ni * torch.eye(nu, device=device)
                S = torch.eye(nu, ny, device=device)
            elif (self.mode == "output_p"):
                assert nu == ny, "Input and Output dimensions must be equal"
                Q = -2*rho * torch.eye(nu, device=device)
                R = torch.zeros(nu, device=device)
                S = torch.eye(nu, ny, device=device)
            else:
                raise NameError("The inserted mode is not valid. Please write 'c', 'rl2', 'input_p' or 'output_p'. :(")
            self.nx = nx
            self.nu = nu
            self.ny = ny
            self.nq = nq

            self.sys = _ImplicitQSRNetwork(nx, ny, nu, nq, sigma, epsilon,
                                           S=S, Q=Q, R=R, gamma=gamma, device=device,
                                           bias=bias, alpha=alpha, feedthrough=feedthrough)

    def _frame(self):
        self.sys._frame()  # type: ignore

    def forward(self, u, x):
        dx, y = self.sys(x, u)
        return dx, y

    def _check(self):
        self.sys._check()  # type: ignore

    def get_obs_size(self):
        return self.nb  # Taille des u pour estimer l'état

    def clone(self):
        copy = type(self)(self.nx, self.ny, self.nu, self.nq,
                          self.sigma, self.epsilon, self.mode,
                          self.gamma, self.device, self.bias,
                          self.ni, self.rho, self.alpha, self.linear_output,
                          self.feedthrough, self.str_save)
        copy.load_state_dict(self.state_dict())
        return copy