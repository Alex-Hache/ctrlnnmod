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

    def __init__(self, u: np.ndarray, y: np.ndarray, ts: float = 1, x=None, nx=None,
                 x_trainable: bool = False, scaled: bool = False) -> None:
        super(Experiment, self).__init__()
        self.u = Tensor(u)
        self.y = Tensor(y)
        self.ts = ts
        self.nu = u.shape[1]
        self.ny = y.shape[1]
        self.scaled = scaled

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

        # Si scaled est True, on normalise directement les données
        if self.scaled:
            self._compute_scaling_factors()
            self._apply_scaling()


    def _compute_scaling_factors(self):
        """Calcule les facteurs d'échelle"""
        self.scaling_factors = {
            'u': {
                'min': self.u.min(dim=0).values,
                'max': self.u.max(dim=0).values
            },
            'y': {
                'min': self.y.min(dim=0).values,
                'max': self.y.max(dim=0).values
            },
            'x': {
                'min': self.x.min(dim=0).values,
                'max': self.x.max(dim=0).values
            }
        }

    def _apply_scaling(self):
        """Applique la normalisation aux données"""

        
        for key in ['u', 'y', 'x']:
            data = getattr(self, key)
            min_vals = self.scaling_factors[key]['min']
            max_vals = self.scaling_factors[key]['max']
            
            # Clone des données pour préserver requires_grad si nécessaire
            scaled_data = data.clone()
            
            # Traitement colonne par colonne
            for i in range(data.shape[1]):
                denominateur = max_vals[i] - min_vals[i]
                if denominateur == 0:  # Variable constante
                    scaled_data[:, i] = data[:, i] - min_vals[i]
                else:
                    scaled_data[:, i] = (data[:, i] - min_vals[i]) / denominateur
                    
            # Préservation de requires_grad si nécessaire
            if getattr(data, 'requires_grad', False):
                scaled_data.requires_grad = True
                
            setattr(self, key, scaled_data)


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

    def denormalize(self, u=None, y=None, x=None):
        """Dénormalise les données si nécessaire"""
        if not self.scaled:
            return u, y, x
            
        results = []
        for data, key in zip([u, y, x], ['u', 'y', 'x']):
            if data is not None:
                factors = self.scaling_factors[key]
                denorm_data = data * (factors['max'] - factors['min']) + factors['min']
                results.append(denorm_data)
            else:
                results.append(None)
                
        return results
    
    def get_data(self, idx=None, unscaled=False):
        '''
        Return the experiment values up to the idx index if idx is not None
        Args:
            idx: Optional[int] - Index jusqu'auquel récupérer les données
            unscaled: bool - Si True et si scaled=True dans l'initialisation, 
                            retourne les données dénormalisées
        Returns:
            Tuple[Tensor, Tensor, Tensor] - (u, y, x) normalisés ou non
        '''
        if idx is not None:
            if idx >= self.n_samples:
                idx = self.n_samples - 1
            u, y, x = self.u[:idx, :], self.y[:idx, :], self.x[:idx, :]
        else:
            u, y, x = self.u, self.y, self.x

        if self.scaled and unscaled:
            u, y, x = self.denormalize(u, y, x) 
        return u.numpy(), y.numpy(), x.detach().numpy()
    
    def plot(self, idx=None, unscaled=False):
        """
        Affiche les données de l'expérience jusqu'à l'index idx
        """
        u, y, x = self.get_data(idx, unscaled)
        t = np.linspace(0, (u.shape[0]-1)*self.ts, u.shape[0])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Experiment Data Visualization')
        
        # Plot des entrées
        for i in range(self.nu):
            ax1.plot(t, u[:, i], label=f'u{i+1}')
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

    def __init__(self, exps: List[Experiment], seq_len: int = 1, scaled = False) -> None:
        super(ExperimentsDataset, self).__init__()

        self.experiments = exps
        self.set_seq_len(seq_len)
        self._set_index_map()  # Setting a map for generating indices
        self.n_exp = len(exps)

        self.scaled = scaled

        # Vérification de la cohérence des données
        self._check_consistency()
        
        # Calcul des facteurs d'échelle globaux si nécessaire
        if self.scaled:
            self._compute_global_scaling_factors()
            self._apply_global_scaling()

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

    def _apply_global_scaling(self) -> None:
        """Applique la normalisation globale à toutes les expériences"""
        for exp in self.experiments:
            for key in ['u', 'y', 'x']:
                data = getattr(exp, key)
                min_vals = self.scaling_factors[key]['min']
                max_vals = self.scaling_factors[key]['max']
                
                # Gestion des variables constantes
                denominateur = max_vals - min_vals
                scaled_data = data.clone()
                
                for i in range(data.shape[1]):
                    if denominateur[i] == 0:  # Variable constante
                        scaled_data[:, i] = data[:, i] - min_vals[i]
                    else:
                        scaled_data[:, i] = (data[:, i] - min_vals[i]) / denominateur[i]

                setattr(exp, key, scaled_data)
                if getattr(data, 'requires_grad', True):
                    new_data = getattr(exp, key)
                    new_data.requires_grad_(True)



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

    def _compute_global_scaling_factors(self) -> None:
        """Calcule les facteurs d'échelle globaux pour toutes les expériences"""
        # Initialisation des min/max globaux
        self.scaling_factors = {
            'u': {'min': None, 'max': None},
            'y': {'min': None, 'max': None},
            'x': {'min': None, 'max': None}
        }

        # Calcul des min/max globaux
        for key in ['u', 'y', 'x']:
            all_data = torch.cat([getattr(exp, key) for exp in self.experiments], dim=0)
            self.scaling_factors[key]['min'] = all_data.min(dim=0).values
            self.scaling_factors[key]['max'] = all_data.max(dim=0).values       


    def append(self, exp: Experiment) -> None:
        """Ajoute une nouvelle expérience au dataset"""
        # Vérification de la cohérence
        if self.experiments:
            base_exp = self.experiments[0]
            assert exp.nu == base_exp.nu, "New experiment must have same number of inputs"
            assert exp.ny == base_exp.ny, "New experiment must have same number of outputs"
            assert exp.nx == base_exp.nx, "New experiment must have same number of states"
            assert exp.ts == base_exp.ts, "New experiment must have same sampling time"

        # Si scaled, application de la normalisation à la nouvelle expérience
        if self.scaled:
            self._update_scaling_factors(exp)
            self._apply_scaling_to_experiment(exp)

        self.experiments.append(exp)
        self._set_index_map()
        self.n_exp += 1

    def _update_scaling_factors(self, new_exp: Experiment) -> None:
        """Met à jour les facteurs d'échelle avec une nouvelle expérience"""
        for key in ['u', 'y', 'x']:
            data = getattr(new_exp, key)
            curr_min = self.scaling_factors[key]['min']
            curr_max = self.scaling_factors[key]['max']
            
            new_min = torch.minimum(curr_min, data.min(dim=0).values)
            new_max = torch.maximum(curr_max, data.max(dim=0).values)
            
            self.scaling_factors[key]['min'] = new_min
            self.scaling_factors[key]['max'] = new_max

    def _apply_scaling_to_experiment(self, exp: Experiment) -> None:
        """Applique la normalisation à une seule expérience"""
        for key in ['u', 'y', 'x']:
            data = getattr(exp, key)
            min_vals = self.scaling_factors[key]['min']
            max_vals = self.scaling_factors[key]['max']
            
            denominateur = max_vals - min_vals
            scaled_data = data.clone()
            
            for i in range(data.shape[1]):
                if denominateur[i] == 0:
                    scaled_data[:, i] = data[:, i] - min_vals[i]
                else:
                    scaled_data[:, i] = (data[:, i] - min_vals[i]) / denominateur[i]
            
            setattr(exp, key, scaled_data)


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