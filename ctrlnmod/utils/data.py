import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import matplotlib.pyplot as plt
import math
import torch
from typeguard import typechecked
from typing import (
    List,
    TypeVar,
)
T_co = TypeVar('T_co', covariant=True)
import lightning as pl


@typechecked
class Experiment(Dataset):
    """
    This class is composed of the following attributes :
        * u : input data
        * y : ouyput (measurement) data
        * x : state data -- optional but necessary for training state-space models
    """

    def __init__(self, u: np.ndarray, y: np.ndarray, ts: float = 1, x=None, nx=None, x_trainable: bool = False) -> None:
        super(Experiment, self).__init__()
        self.u = Tensor(u)
        self.y = Tensor(y)
        self.ts = ts
        self.nu = u.shape[1]
        self.ny = y.shape[1]

        assert u.shape[0] == y.shape[0], ValueError(
            f"u and y do not have the same number or samples found {u.shape[0]} vs {y.shape[0]}")
        self.n_samples = u.shape[0]
        if x_trainable and x is not None:
            raise ValueError(
                "A trainable x is incompatible with giving x from experiments")
        if x is not None:
            self.x = Tensor(x)
            self.nx = x.shape[1]
            self.x_trainable = x_trainable
        elif x is None and nx is not None:
            self.nx = nx
            self.x_trainable = x_trainable
            self.x = torch.zeros((self.n_samples, self.nx),
                                 requires_grad=x_trainable)
        else:
            raise ValueError("Please specify a size for state vector")

    def __getitem__(self, idx, seq_len):
        """
            Returns a Tuple composed of u_idx, y_idx, x_idx
            from idx to idx+seq_len
        """
        return (self.u[idx:idx + seq_len, :], self.y[idx:idx + seq_len, :], self.x[idx:idx + seq_len, :], self.x[idx, :])

    def __len__(self):
        return self.n_samples

    def __repr__(self):
        return f'Experiment of length: {self.n_samples} nu={self.nu} ny={self.ny} dt={self.ts}'

    def __str__(self) -> str:
        return f"Exp_nu={self.nu}_ny={self.ny}_dt={self.ts}"

    def get_data(self, idx=None):
        '''
            Return the experiment values up to the idx index if idx is not None
        '''
        if idx is not None:
            if idx >= self.n_samples:
                idx = self.n_samples - 1
            return self.u[:idx, :], self.y[:idx, :], self.x[:idx, :]
        else:
            return self.u, self.y, self.x


@typechecked
class ExperimentsDataset(Dataset):
    """
    This class implements methods to deal with sequences potentially
    coming from several experiments and from different length.
    """

    def __init__(self, exps: List[Experiment], seq_len: int = 1) -> None:
        super(ExperimentsDataset, self).__init__()

        self.experiments = exps
        self.set_seq_len(seq_len)
        self._set_index_map()  # Setting a map for generating indices
        self.n_exp = len(exps)

    def __getitem__(self, index: int):
        exp_index, sample_index = self.index_map[index]
        experiment = self.experiments[exp_index]
        return experiment.__getitem__(sample_index, self.seq_len)

    def __len__(self):
        return self.n_exp_samples_avl

    def _set_index_map(self):
        index, n_exp_samples = 0, 0
        index_map = {}
        # First iterating over experiments
        for id_exp, exp in enumerate(self.experiments):
            # Then iterating over samples
            for id_sample in range(len(exp) - self.seq_len + 1):
                index_map[index] = (id_exp, id_sample)
                index += 1
            n_exp_samples += len(exp)
        self.index_map = index_map
        self.n_exp_samples_avl = index  # Number of total data points available
        self.n_exp_samples = n_exp_samples

    def append(self, exp: Experiment) -> None:
        self.experiments.append(exp)
        self._set_index_map()  # Update index map
        self.n_exp += 1  # Updater number of experiments

        # Assert if all experiments have same sampling period
        ts_list = [experiment.ts for experiment in self.experiments]

        if not all([ts == ts_list[0] for ts in ts_list]):
            raise NotImplementedError("All sampling times must be identical")

        nx_list = [experiment.nx for experiment in self.experiments]
        if not all([nx == nx_list[0] for nx in nx_list]):
            raise (NotImplementedError("All state orders must be identical"))

    def __repr__(self) -> str:
        return f"Dataset of {self.n_exp} experiments -- Total number of samples : {self.n_exp_samples}"

    @typechecked
    def set_seq_len(self, seq_len: int):
        if seq_len == -1:
            # If -1 all samples are taken only the smallest one
            seq_len = min([len(exp) for exp in self.experiments])
        max_length = max([len(exp) for exp in self.experiments])
        if seq_len > max_length or seq_len <= 0:
            raise ValueError(
                f"Invalid sequence length : longest experiment is length {max_length} samples asked {seq_len}")
        self.seq_len = seq_len
        self._set_index_map()  # Update index map

    def plot(self, figsize=(15, 10), max_exp_to_plot=4):
        num_experiments = len(self.experiments)
        num_figures = min(max_exp_to_plot, math.ceil(num_experiments / 4))

        for fig_num in range(num_figures):
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(
                f'Experiments {fig_num*4+1} to {min((fig_num+1)*4, num_experiments)}')
            axs = axs.flatten()

            for i in range(4):
                exp_idx = fig_num * 4 + i
                if exp_idx >= num_experiments:
                    break

                exp = self.experiments[exp_idx]
                ax = axs[i]

                u, y, _ = exp.get_data()
                time = torch.arange(0, len(u)) * exp.ts

                for j in range(exp.nu):
                    ax.plot(time, u[:, j], label=f'u{j+1}')
                for j in range(exp.ny):
                    ax.plot(time, y[:, j], label=f'y{j+1}', linestyle='--')

                ax.set_title(f'Experiment {exp_idx + 1}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plt.show()


class ExperimentsDataModule(pl.LightningDataModule):
    def __init__(self, train_set: ExperimentsDataset, val_set: ExperimentsDataset, 
                 batch_size: int = 32, num_workers: int = 0, max_idx_val: int = None):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_idx_val = max_idx_val
        
    def setup(self, stage=None):
        if self.max_idx_val is None:
            self.max_idx_val = min(exp.n_samples for exp in self.val_set.experiments)

    def val_dataloader(self):
        batch_u = torch.stack([exp.u[:self.max_idx_val] for exp in self.val_set.experiments])
        batch_y_true = torch.stack([exp.y[:self.max_idx_val] for exp in self.val_set.experiments])
        batch_x0 = torch.stack([exp.x[0] for exp in self.val_set.experiments])
        
        dataset = torch.utils.data.TensorDataset(batch_u, batch_y_true, batch_x0)
        return torch.utils.data.DataLoader(dataset, batch_size=len(self.val_set.experiments))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )