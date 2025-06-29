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
    Optional
)
T_co = TypeVar('T_co', covariant=True)
import lightning as pl

@typechecked
class Experiment(Dataset):
    """
    This class represents a single experiment with inputs (u), outputs (y),
    and optionally states (x) and disturbances (d). It is designed to handle
    time-series data, where each experiment can have a different number of samples.

    Attributes:
        u (Tensor): Input data of shape (n_samples, nu).
        y (Tensor): Output data of shape (n_samples, ny).
        ts (float): Sampling time.
        x (Tensor, optional): State data of shape (n_samples, nx).
        nu (int): Number of inputs.
        ny (int): Number of outputs.
        nx (int, optional): Number of states.
        n_samples (int): Number of samples in the experiment.
        x_trainable (bool): Whether the state vector is trainable.
        d (Tensor, optional): Disturbance data of shape (n_samples, nd).

    Methods:
        __getitem__(idx, seq_len): Returns a tuple of (u, y, x, x0) for the given index and sequence length.
        __len__(): Returns the number of samples in the experiment.
        denormalize(u=None, y=None, x=None, scaler=None): Denormalizes the data if a scaler is provided.
        get_data(idx=None, unscaled=False, scaler=None): Returns the experiment data up to the specified index.
        plot(idx=None, unscaled=False, scaler=None): Plots the experiment data.

    Args:
        u (np.ndarray): Input data of shape (n_samples, nu).
        y (np.ndarray): Output data of shape (n_samples, ny).
        ts (float): Sampling time.
        x (np.ndarray, optional): State data of shape (n_samples, nx). Defaults to None.
        nx (int, optional): Number of states. Must match x.shape[1] if x is provided. Defaults to None.
        x_trainable (bool): Whether the state vector is trainable. Defaults to False.
        d (np.ndarray, optional): Disturbance data of shape (n_samples, nd). Defaults to None.

    Raises:
        ValueError: If u and y do not have the same number of samples.
        ValueError: If x is provided but x_trainable is True.
        ValueError: If x is None and nx is not provided.
        ValueError: If seq_len is invalid.
    """

    def __init__(self, u: np.ndarray, y: np.ndarray, ts: float = 1, x=None, nx=None,
                 x_trainable: bool = False, d=None) -> None:
        super(Experiment, self).__init__()
        self.u = Tensor(u)
        self.y = Tensor(y)
        self.d = Tensor(d) if d is not None else None
        self.ts = ts
        self.nu = u.shape[1]
        self.ny = y.shape[1]
        self.nd = d.shape[1] if d is not None else 0
        if d is not None and self.d.shape[0] != u.shape[0]:
            raise ValueError(
                f"u and d do not have the same number of samples found {u.shape[0]} vs {self.d.shape[0]}")

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

    def __getitem__(self, idx, seq_len): # type: ignore
        """
            Returns a Tuple composed of u_idx, y_idx, x_idx and optionally d_idx
            from idx to idx+seq_len
        """
        return (self.u[idx:idx + seq_len, :], self.y[idx:idx + seq_len, :], self.x[idx:idx + seq_len, :], self.x[idx, :],
                self.d[idx:idx + seq_len, :] if self.d is not None else None)

    def __len__(self):
        return self.n_samples

    def __repr__(self) -> str:
        repr = f"Experiment with nu={self.nu}, ny={self.ny}, nx={self.nx}, n_samples={self.n_samples}, ts={self.ts}"
        if self.d is not None:
            repr += f", nd={self.nd}"
        if self.x_trainable:
            repr += ", x_trainable=True"
        else:
            repr += ", x_trainable=False"
        return repr

    def __str__(self) -> str:
        return self.__repr__()

    def denormalize(self, u: Optional[Tensor] = None, y:Optional[Tensor] =None, x: Optional[Tensor]=None, 
                    d: Optional[Tensor]=None, 
                    scaler =None):
        """Dénormalise les données si un scaler est fourni"""
        if scaler is None:
            return u, y, x, d
            
        results = []
        temp_exp = Experiment(
            u=u.numpy() if u is not None else np.zeros((1, self.nu)),
            y=y.numpy() if y is not None else np.zeros((1, self.ny)),
            x=x.numpy() if x is not None else np.zeros((1, self.nx)),
            d=d.numpy() if d is not None else np.zeros((1, self.nd)) if self.nd > 0 else None,
            ts=self.ts
        )
        
        scaler.inverse_transform(temp_exp)
        
        return (
            temp_exp.u if u is not None else None,
            temp_exp.y if y is not None else None,
            temp_exp.x if x is not None else None,
            temp_exp.d if d is not None else None
        )
    
    def get_data(self, idx=None, unscaled=False, scaler=None):
        '''
        Return the experiment values up to the idx index if idx is not None

        Args:
            idx: Optional[int] - Index jusqu'auquel récupérer les données
            unscaled: bool - Si True et si un scaler est fourni, retourne les données dénormalisées
            scaler: Optional[BaseScaler] - Scaler pour la dénormalisation

        Returns:
            Tuple[Tensor, Tensor, Tensor] - (u, y, x) normalisés ou non
        '''
        if idx is not None:
            if idx >= self.n_samples:
                idx = self.n_samples
            u, y, x, d = self.u[:idx, :], self.y[:idx, :], self.x[:idx, :], self.d[:idx, :] if self.d is not None else None
        else:
            u, y, x, d = self.u, self.y, self.x, self.d if self.d is not None else None

        if unscaled and scaler is not None:
            u, y, x, d = self.denormalize(u, y, x, scaler) 
        
        return (
            u.numpy(), 
            y.numpy(),
            x.detach().numpy(),
            d.numpy() if d is not None else None
        )
    
    def plot(self, idx=None, unscaled=False, scaler = None):
        """
        Affiche les données de l'expérience jusqu'à l'index idx
        """
        u, y, x, d = self.get_data(idx, unscaled, scaler)
        t = np.linspace(0, (u.shape[0]-1)*self.ts, u.shape[0])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Experiment Data Visualization')
        
        # Plot des entrées
        for i in range(self.nu):
            ax1.plot(t, u[:, i], label=f'u{i+1}')
        if d is not None:
            for i in range(self.nd):
                ax1.plot(t, d[:, i], label=f'd{i+1}')
        ax1.set_ylabel('Inputs')
        ax1.grid(True)
        ax1.legend()
        
        # Plot des sorties
        for i in range(self.ny):
            ax2.plot(t, y[:, i], label=f'y{i+1}')
        ax2.set_ylabel('Outputs')
        ax2.grid(True)
        ax2.legend()
        
        # Plot des états
        for i in range(self.nx):
            ax3.plot(t, x[:, i], label=f'x{i+1}')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('States')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2, ax3)

@typechecked
class ExperimentsDataset(Dataset):
    """
    This class implements methods to deal with sequences potentially
    coming from several experiments and from different length.
    """

    def __init__(self, exps: List[Experiment], seq_len: int = 1, scaler = None) -> None:
        super(ExperimentsDataset, self).__init__()

        self.experiments = exps
        self.set_seq_len(seq_len)
        self._set_index_map()  # Setting a map for generating indices
        self.n_exp = len(exps)

        self.scaler = scaler

        # Vérification de la cohérence des données
        self._check_consistency()
        
        # Calcul des facteurs d'échelle globaux si nécessaire
        if self.scaler is not None:
            self.scaler.fit(self.experiments)
            for exp in self.experiments:
                self.scaler.transform(exp)

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


    def _check_consistency(self) -> None:
        """Vérifie la cohérence des expériences"""
        if not self.experiments:
            return

        base_exp = self.experiments[0]
        for exp in self.experiments[1:]:
            assert exp.nu == base_exp.nu, "All experiments must have same number of inputs"
            assert exp.ny == base_exp.ny, "All experiments must have same number of outputs"
            assert exp.nx == base_exp.nx, "All experiments must have same number of states"
            assert exp.ts == base_exp.ts, "All experiments must have same sampling time" 
            assert exp.nd == base_exp.nd, "All experiments must have same number of disturbances"
        self.ts = base_exp.ts


    def append(self, exp: Experiment) -> None:
        """Ajoute une nouvelle expérience au dataset"""
        # Vérification de la cohérence
        if self.experiments:
            base_exp = self.experiments[0]
            assert exp.nu == base_exp.nu, "New experiment must have same number of inputs"
            assert exp.ny == base_exp.ny, "New experiment must have same number of outputs"
            assert exp.nx == base_exp.nx, "New experiment must have same number of states"
            assert exp.ts == base_exp.ts, "New experiment must have same sampling time"
            assert exp.nd == base_exp.nd, "All experiments must have same number of disturbances"

        # Si scaled, application de la normalisation à la nouvelle expérience
        # Application du scaler si présent
        if self.scaler is not None and self.scaler.is_fitted:
            self.scaler.transform(exp)

        self.experiments.append(exp)
        self._set_index_map()
        self.n_exp += 1

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

    def plot(self, figsize=(15, 10), max_exp_to_plot=4, unscaled=False):
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

                u, y, _, d = exp.get_data(unscaled=unscaled, scaler=self.scaler)
                time = torch.arange(0, len(u)) * exp.ts

                for j in range(exp.nu):
                    ax.plot(time, u[:, j], label=f'u{j+1}')
                    if d is not None:
                        ax.plot(time, d[:, j], label=f'd{j+1}', linestyle=':')
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
                 batch_size: int = 32, num_workers: int = 0, max_idx_val: Optional[int] = None):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_idx_val = max_idx_val
        self.ts = train_set.ts
        
    def setup(self, stage=None):
        if self.max_idx_val is None:
            self.max_idx_val = min(exp.n_samples for exp in self.val_set.experiments)

    def val_dataloader(self, unscaled=False):
        if unscaled and self.val_set.scaler is not None:
            # Créer une copie temporaire des expériences
            temp_experiments = []
            for exp in self.val_set.experiments:
                temp_exp = Experiment(
                    u=exp.u.clone().numpy(),
                    y=exp.y.clone().numpy(),
                    x=exp.x.clone().numpy(),
                    d=exp.d.clone().numpy() if exp.d is not None else None,
                    ts=exp.ts
                )
                self.val_set.scaler.inverse_transform(temp_exp)
                temp_experiments.append(temp_exp)
        else:
            temp_experiments = self.val_set.experiments

        batch_u = torch.stack([exp.u[:self.max_idx_val] for exp in temp_experiments])
        batch_y_true = torch.stack([exp.y[:self.max_idx_val] for exp in temp_experiments])
        batch_x0 = torch.stack([exp.x[0] for exp in temp_experiments])
        batch_d = torch.stack([exp.d[:self.max_idx_val] for exp in temp_experiments]) if temp_experiments[0].d is not None else None
        
        if batch_d is not None:
            dataset = torch.utils.data.TensorDataset(batch_u, batch_y_true, batch_x0, batch_d)
        else:
            dataset = torch.utils.data.TensorDataset(batch_u, batch_y_true, batch_x0)
        return torch.utils.data.DataLoader(dataset, batch_size=len(self.val_set.experiments))

    def train_dataloader(self, unscaled=False):
        if unscaled:
            # Créer un dataloader temporaire avec données dénormalisées
            temp_dataset = ExperimentsDataset(
                [Experiment(
                    u=exp.u.clone().numpy(),
                    y=exp.y.clone().numpy(),
                    x=exp.x.clone().numpy(),
                    d=exp.d.clone().numpy() if exp.d is not None else None,
                    ts=exp.ts
                ) for exp in self.train_set.experiments],
                seq_len=self.train_set.seq_len
            )
            if self.train_set.scaler is not None:
                for exp in temp_dataset.experiments:
                    self.train_set.scaler.inverse_transform(exp)
            dataset = temp_dataset
        else:
            dataset = self.train_set
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )