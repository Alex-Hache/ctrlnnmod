import torch
import torch.nn as nn
from torch import Tensor
import time
import numpy as np
import os
import geotorch_custom.parametrize as P


class Simulator(nn.Module):
    '''
    This class implements the base class on which simulation model is built
    depending on the integration method only the forward method changes.
    '''

    def __init__(self, ss_model, ts, nb: int = 1) -> None:
        super(Simulator, self).__init__()

        self.ss_model = ss_model
        self.ts = ts
        self.nx = self.ss_model.nx
        self.nb = nb  # Number of past inputs to estimate current state
        self.save_path = os.getcwd()

    def __str__(self) -> str:
        return f"{str(self.ss_model)}"

    def set_save_path(self, path):
        if path is not None:
            self.save_path = path

    def save(self) -> None:
        torch.save(self.state_dict(), self.save_path + '/model.pkl')

    @staticmethod
    def load(path):
        return torch.load(path, weights_only=False)

    def get_obs_size(self):
        return self.nb  # Number of observations need for state estimation

    def simulate(self, u: Tensor, x0):
        '''
            Inputs :
                * u : Input tensor of shape (n_exp, n_samples, nu)
                * x0 : initial condition of shape (nx)
        '''
        start_time = time.time()
        with torch.no_grad():
            x_sim, y_sim = self(u[None, :, :], x0[None, :])  # Appel à forward
            x_sim.squeeze_(0)
            y_sim.squeeze_(0)
            sim_time = time.time() - start_time
            print(f"\nSim time: {sim_time:.2f}")
        return x_sim, y_sim

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

    def check_(self):
        if hasattr(self.ss_model, 'check_'):
            return self.ss_model.check_()
        else:
            return None


class RK4Simulator(Simulator):
    def __init__(self, ss_model, ts):
        super(RK4Simulator, self).__init__(ss_model=ss_model, ts=ts)

    def __repr__(self):
        return f"RK4 integrator : ts={self.ts}"

    def __str__(self) -> str:
        return f"RK4_{str(self.ss_model)}"

    def forward(self, u_batch, x0_batch=torch.zeros(1)):
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

        with P.cached():  # Useful if there is any parameterized models
            for u_step in u_batch.split(1, dim=1):  # i in range(seq_len):

                u_step = u_step.squeeze(1)
                # x_step = x_step.squeeze(0)

                X_sim_list += [x_step]

                dt2 = self.ts / 2.0
                k1, y_step = self.ss_model(u_step, x_step)
                Y_sim_list += [y_step]

                k2_dx, _ = self.ss_model(u_step, x_step + dt2 * k1)
                k3_dx, _ = self.ss_model(u_step, x_step + dt2 * k2_dx)
                k4_dx, _ = self.ss_model(u_step, x_step + self.ts * k3_dx)

                dx = self.ts / 6.0 * (k1 + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx)
                x_step = x_step + dx

            X_sim = torch.stack(X_sim_list, 1)
            Y_sim = torch.stack(Y_sim_list, 1)
        return X_sim, Y_sim

    def clone(self):
        copy_ss = self.ss_model.clone()  # State-space model module must have a clone function
        copy = type(self)(copy_ss, self.ts)
        copy.load_state_dict(self.state_dict())
        return copy


class Sim_discrete(Simulator):
    def __init__(self, ss_model, ts=1):
        super(Sim_discrete, self).__init__(ss_model=ss_model, ts=ts)

    def clone(self):
        copy_ss = self.ss_model.clone()  # State-space model module must have a clone function
        copy = type(self)(copy_ss, self.ts)
        copy.load_state_dict(self.state_dict())
        return copy

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
