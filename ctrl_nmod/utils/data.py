import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import torch
from typing import (
    List,
    TypeVar,
)
T_co = TypeVar('T_co', covariant=True)


class Experiment(Dataset):
    """
    This class is composed of the following attributes :
        * u : input data
        * y : ouyput (measurement) data
        * x : state data -- optional but necessary for training with subsequences
    """
    def __init__(self, u: np.ndarray, y: np.ndarray, ts: float = 1, x=None, nx=None, x_trainable: bool = False) -> None:
        super(Experiment, self).__init__()
        self.u = Tensor(u)
        self.y = Tensor(y)
        self.ts = ts
        self.nu = u.shape[1]
        self.ny = y.shape[1]

        assert u.shape[0] == y.shape[0], f"u and y do not have the same number or samples found {u.shape[0]} vs {y.shape[0]}"
        self.n_samples = u.shape[0]
        if x_trainable and x is not None:
            raise ValueError("A trainable x is incompatible with giving x from experiments")
        if x is not None:
            self.x = Tensor(x)
            self.nx = x.shape[1]
            self.x_trainable = x_trainable
        elif x is None and nx is not None:
            self.nx = nx
            self.x_trainable = x_trainable
            self.x = torch.zeros((self.n_samples, self.nx), requires_grad=x_trainable)
        else:
            raise ValueError("Please specify a size for state vector")

    def __getitem__(self, idx, seq_len):
        """
            Returns a Tuple composed of u_idx, y_idx, x_idx
            from idx to idx+seq_len
        """
        return (self.u[idx:idx+seq_len, :], self.y[idx:idx+seq_len, :], self.x[idx:idx+seq_len, :], self.x[idx, :])

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
                idx = self.n_samples-1
            return self.u[:idx, :], self.y[:idx, :], self.x[:idx, :]
        else:
            return self.u, self.y, self.x


class ExperimentsDataset(Dataset):
    """
    This class implements methods to deal with sequences potentially
    coming from several experiments.
    """

    def __init__(self, exps: List[Experiment], seq_len: int = 1) -> None:
        super(ExperimentsDataset, self).__init__()

        self.experiments = exps
        self._set_index_map()  # Setting a map for generating indices
        self.n_exp = len(exps)
        self.seq_len = seq_len

    def __getitem__(self, index):
        exp_index, sample_index = self.index_map[index]
        experiment = self.experiments[exp_index]
        return experiment.__getitem__(sample_index, self.seq_len)

    def __len__(self):
        return self.n_exp_samples - self.seq_len

    def _set_index_map(self):
        index = 0
        index_map = {}
        for id_exp, exp in enumerate(self.experiments):  # First iterating over experiments
            for id_sample in range(len(exp)):  # Then iterating over samples
                index_map[index] = (id_exp, id_sample)
                index += 1
        self.index_map = index_map
        self.n_exp_samples = index + 1  # Number of total data points

    def append(self, exp):
        self.experiments.append(exp)
        self._set_index_map()  # Update index map
        self.n_exp += 1  # Updater number of experiments

    def __repr__(self) -> str:
        return f"Dataset of {self.n_exp} experiments -- Total number of samples : {len(self)}"

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
