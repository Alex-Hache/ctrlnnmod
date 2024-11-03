import torch
import torch.nn as nn
from torch import Tensor
import time
import numpy as np
import os
import geotorch_custom.parametrize as P

class Simulator(nn.Module):
    def __init__(self, ss_model: nn.Module, ts, nb: int = 1) -> None:
        super(Simulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts  # sampling time
        self.nx = self.ss_model.nx  # State order
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
        return self.nb  # Number of observations needed for state estimation

    def simulate(self, u: Tensor, x0: Tensor):
        start_time = time.time()
        with torch.no_grad():
            x_sim, y_sim = self(u[None, :, :], x0[None, :])  # Call to forward
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
        return np.array(weights, dtype=object), np.array(biases, dtype=object)

    def _frame(self):
        return self.ss_model._frame()

    def check(self):
        if hasattr(self.ss_model, 'check'):
            return self.ss_model.check()
        else:
            return True, {}

class RK4Simulator(Simulator):
    def __init__(self, ss_model, ts):
        super(RK4Simulator, self).__init__(ss_model=ss_model, ts=ts)

    def __repr__(self):
        return f"RK4 integrator : ts={self.ts}"

    def __str__(self) -> str:
        return f"RK4_{str(self.ss_model)}"

    def forward(self, u_batch, x0_batch=torch.zeros(1)):
        X_sim_list = []
        Y_sim_list = []
        x_step = x0_batch

        with P.cached(), self.ss_model._frame_cache.cache_frame():
            for u_step in u_batch.split(1, dim=1):
                u_step = u_step.squeeze(1)
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

    @classmethod
    def discretize(cls, A, h):
        """
        Discretize matrix A using the RK4 method and return the Ad matrix.
        """
        I = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        hA = h * A
        k1 = hA
        k2 = hA @ (I + 0.5 * k1)
        k3 = hA @ (I + 0.5 * k2)
        k4 = hA @ (I + k3)
        return I + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    def clone(self):
        copy_ss = self.ss_model.clone()
        copy = type(self)(copy_ss, self.ts)
        copy.load_state_dict(self.state_dict())
        return copy

class RK45Simulator(Simulator):
    def __init__(self, ss_model, ts):
        super(RK45Simulator, self).__init__(ss_model=ss_model, ts=ts)
        self.x_f = None

    def __repr__(self):
        return f"RK45 integrator : ts={self.ts}"

    def __str__(self) -> str:
        return f"RK45_{str(self.ss_model)}"
    
    def forward(self, u_batch, x0_batch=torch.zeros(1)):
        X_sim_list = []
        Y_sim_list = []
        x_step = x0_batch

        with P.cached(), self.ss_model._frame_cache.cache_frame():
            for u_step in u_batch.split(1, dim=1):
                u_step = u_step.squeeze(1)
                X_sim_list.append(x_step)
                
                dt = self.ts
                k1, y_step = self.ss_model(u_step, x_step)
                Y_sim_list.append(y_step)
                
                k2, _ = self.ss_model(u_step, x_step + dt/4 * k1)
                k3, _ = self.ss_model(u_step, x_step + dt/32 * (3*k1 + 9*k2))
                k4, _ = self.ss_model(u_step, x_step + dt/2197 * (1932*k1 - 7200*k2 + 7296*k3))
                k5, _ = self.ss_model(u_step, x_step + dt * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
                k6, _ = self.ss_model(u_step, x_step + dt * (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
                
                # Solution d'ordre 5
                dx = dt * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
                x_step = x_step + dx

            X_sim = torch.stack(X_sim_list, 1)
            Y_sim = torch.stack(Y_sim_list, 1)
            self.x_f = x_step
        return X_sim, Y_sim

    def clone(self):
        copy_ss = self.ss_model.clone()
        copy = type(self)(copy_ss, self.ts)
        copy.load_state_dict(self.state_dict())
        return copy

    @classmethod
    def discretize(cls, A, h):
        """
        Discretize matrix A using the RK45 method and return the Ad matrix.
        """
        I = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        hA = h * A
        k1 = hA
        k2 = hA @ (I + 0.25 * k1)
        k3 = hA @ (I + 3/32 * k1 + 9/32 * k2)
        k4 = hA @ (I + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
        k5 = hA @ (I + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
        k6 = hA @ (I - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
        
        return I + (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)

class Sim_discrete(Simulator):
    def __init__(self, ss_model, ts=1):
        super(Sim_discrete, self).__init__(ss_model=ss_model, ts=ts)

    def clone(self):
        copy_ss = self.ss_model.clone()
        copy = type(self)(copy_ss, self.ts)
        copy.load_state_dict(self.state_dict())
        return copy

    def forward(self, u_batch, x0_batch=torch.Tensor(1)):
        X_sim_list = []
        Y_sim_list = []
        x_step = x0_batch
        for u_step in u_batch.split(1, dim=1):
            u_step = u_step.squeeze(1)
            X_sim_list += [x_step]

            x_step, y_step = self.ss_model(u_step, x_step)
            Y_sim_list += [y_step]

        X_sim = torch.stack(X_sim_list, 1)
        Y_sim = torch.stack(Y_sim_list, 1)
        return X_sim, Y_sim