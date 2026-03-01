from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import torch
from typeguard import typechecked
from ..utils import Experiment

class BaseScaler(ABC):
    """Abstract class for all linear transformations"""
    
    def __init__(self, feature_names: List[str] = ['u', 'y', 'x', 'd']):
        self.feature_names = feature_names
        self.is_fitted = False
        self.transform_matrices: Dict[str, torch.Tensor] = {}
        self.transform_biases: Dict[str, torch.Tensor] = {}
        
    @abstractmethod
    def fit(self, experiments: List['Experiment']) -> None:
        """Compute transformation parameters from data/"""
        pass
    
    def transform(self, experiment: 'Experiment') -> None:
        """Apply transform to an experiment"""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
            
        for key in self.feature_names:
            if hasattr(experiment, key):
                data = getattr(experiment, key)
                T = self.transform_matrices[key]
                b = self.transform_biases[key]
                
                # Applying linear transformation
                transformed_data = torch.matmul(data, T.T) + b
                
                # Store requires_grad value
                if getattr(data, 'requires_grad', False):
                    transformed_data.requires_grad_(True)
                    
                setattr(experiment, key, transformed_data)
                
    def inverse_transform(self, experiment: 'Experiment') -> None:
        """Apply the inverse transform to an experiment"""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
            
        for key in self.feature_names:
            if hasattr(experiment, key):
                data = getattr(experiment, key)
                T = self.transform_matrices[key]
                b = self.transform_biases[key]
                
                # Apply the inverse transformation
                T_inv = torch.linalg.inv(T)
                original_data = torch.matmul(data - b, T_inv.T)

                # Preserve gradient if needed
                if getattr(data, 'requires_grad', False):
                    original_data.requires_grad_(True)
                    
                setattr(experiment, key, original_data)



    def save_parameters(self) -> Dict:
        """Save scaler transformation parameters"""
        if not self.is_fitted:
            raise RuntimeError("Cannot save parameters of unfitted scaler")
            
        params = {
            'feature_names': self.feature_names,
            'transform_matrices': {k: v.clone() for k, v in self.transform_matrices.items()},
            'transform_biases': {k: v.clone() for k, v in self.transform_biases.items()},
            'scaler_type': self.__class__.__name__,
            'is_fitted': self.is_fitted
        }
        
        
        if isinstance(self, MinMaxScaler):
            params.update({
                'data_min': {k: v.clone() for k, v in self.data_min.items()},
                'data_max': {k: v.clone() for k, v in self.data_max.items()},
                'feature_range': self.feature_range
            })
        elif isinstance(self, StandardScaler):
            params.update({
                'means': {k: v.clone() for k, v in self.means.items()},
                'stds': {k: v.clone() for k, v in self.stds.items()}
            })
        elif isinstance(self, CustomTScaler):
            # For CustomTScaler, matrices and biases are already saved in transform and inverse_transform
            pass
        
        return params
    
    @classmethod
    def load_parameters(cls, params: Dict) -> 'BaseScaler':
        """Load parameters into a new scaler instance."""
        scaler_type = params['scaler_type']
        if scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler(
                feature_names=params['feature_names'],
                feature_range=params['feature_range']
            )
            scaler.data_min = params['data_min']
            scaler.data_max = params['data_max']
        elif scaler_type == 'StandardScaler':
            scaler = StandardScaler(feature_names=params['feature_names'])
            scaler.means = params['means']
            scaler.stds = params['stds']
        elif scaler_type == 'CustomTScaler':
            scaler = CustomTScaler(
                transform_matrices=params['transform_matrices'],
                transform_biases=params['transform_biases'],
                feature_names=params['feature_names']
            )
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        scaler.transform_matrices = {k: v.clone() for k, v in params['transform_matrices'].items()}
        scaler.transform_biases = {k: v.clone() for k, v in params['transform_biases'].items()}
        scaler.is_fitted = params['is_fitted']
        
        return scaler
    
    def clone(self) -> 'BaseScaler':
        """Create an exact copy of this scaler."""
        params = self.save_parameters()
        return self.load_parameters(params)
    

class MinMaxScaler(BaseScaler):
    """Min-max normalisation scaler."""
    
    def __init__(self, feature_names: List[str] = ['u', 'y', 'x', 'd'],
                 feature_range: tuple = (0, 1)):
        super().__init__(feature_names)
        self.feature_range = feature_range
        self.data_min: Dict[str, torch.Tensor] = {}
        self.data_max: Dict[str, torch.Tensor] = {}
        
    def fit(self, experiments: List['Experiment']) -> None:
        """Compute per-feature min and max across all experiments."""
        for key in self.feature_names:
            if not hasattr(experiments[0], key):
                continue

            # Concatenate data across all experiments
            all_data = torch.cat([getattr(exp, key) for exp in experiments], dim=0)

            # Compute min/max
            self.data_min[key] = all_data.min(dim=0).values
            self.data_max[key] = all_data.max(dim=0).values

            # Build transformation matrix
            data_range = self.data_max[key] - self.data_min[key]
            scale = torch.where(data_range == 0,
                              torch.ones_like(data_range),
                              data_range)

            # Build diagonal transformation matrix
            n_features = all_data.shape[1]
            T = torch.diag(1.0 / scale)
            b = -self.data_min[key] / scale
            
            self.transform_matrices[key] = T
            self.transform_biases[key] = b
            
        self.is_fitted = True

class StandardScaler(BaseScaler):
    """Standardisation scaler (zero mean, unit variance)."""

    def __init__(self, feature_names: List[str] = ['u', 'y', 'x', 'd']):
        super().__init__(feature_names)
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}

    def fit(self, experiments: List['Experiment']) -> None:
        """Compute per-feature mean and standard deviation across all experiments."""
        for key in self.feature_names:
            if not hasattr(experiments[0], key):
                continue

            # Concatenate data across all experiments
            all_data = torch.cat([getattr(exp, key) for exp in experiments], dim=0)

            # Compute mean/std
            self.means[key] = all_data.mean(dim=0)
            self.stds[key] = all_data.std(dim=0)

            # Handle constant features (std == 0)
            self.stds[key] = torch.where(self.stds[key] == 0,
                                       torch.ones_like(self.stds[key]),
                                       self.stds[key])

            # Build diagonal transformation matrix
            n_features = all_data.shape[1]
            T = torch.diag(1.0 / self.stds[key])
            b = -self.means[key] / self.stds[key]
            
            self.transform_matrices[key] = T
            self.transform_biases[key] = b
            
        self.is_fitted = True

class CustomTScaler(BaseScaler):
    """
    Scaler that applies user-defined linear transformation matrices directly.
    For each feature (u, y, x) the transformation is:
        x_transformed = x @ T.T + b
    where T is an invertible square matrix and b is a bias vector.
    """

    def __init__(self, transform_matrices: Dict[str, torch.Tensor],
                 transform_biases: Optional[Dict[str, torch.Tensor]] = None,
                 feature_names: List[str] = ['u', 'y', 'x', 'd']):
        """
        Args:
            transform_matrices: Dict of transformation matrices T for each feature.
            transform_biases: Dict of bias vectors for each feature.
                              If None, biases are initialised to zero.
            feature_names: List of feature names to transform.
        """
        super().__init__(feature_names)

        # Validate that all keys belong to the declared features
        for key in transform_matrices.keys():
            if not (key in feature_names):
                raise ValueError(f"Feature {key} not in feature_names {feature_names}")

        # Validate transformation matrices
        for key, matrix in transform_matrices.items():
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"Matrix for {key} must be a torch.Tensor")

            if not (matrix.dim() == 2):
                raise ValueError(f"Matrix for {key} must be 2D, got {matrix.dim()}D")

            n, m = matrix.shape
            if not (n == m):
                raise ValueError(f"Matrix for {key} must be square, got shape {matrix.shape}")

            det = torch.linalg.det(matrix)
            if not (not torch.isclose(det, torch.tensor(0.0))):
                raise ValueError(f"Matrix for {key} must be invertible, got determinant {det}")

        self.transform_matrices = transform_matrices

        # Initialise or validate biases
        if transform_biases is None:
            self.transform_biases = {
                key: torch.zeros(matrix.shape[0])
                for key, matrix in transform_matrices.items()
            }
        else:
            # Validate provided biases
            for key, bias in transform_biases.items():
                if not (key in transform_matrices):
                    raise ValueError(f"Bias provided for {key} but no matrix found")
                if not isinstance(bias, torch.Tensor):
                    raise ValueError(f"Bias for {key} must be a torch.Tensor")
                if not (bias.dim() == 1):
                    raise ValueError(f"Bias for {key} must be 1D, got {bias.dim()}D")
                if not (bias.shape[0] == transform_matrices[key].shape[0]):
                    raise ValueError(f"Bias dimension {bias.shape[0]} does not match matrix dimension {transform_matrices[key].shape[0]}")
            self.transform_biases = transform_biases

        self.is_fitted = True  # Matrices are provided directly; scaler is ready immediately

    def fit(self, experiments: List['Experiment']) -> None:
        """
        No-op: transformation matrices are already defined.
        Validates that matrix dimensions are consistent with the data.
        """
        for key in self.transform_matrices.keys():
            if not hasattr(experiments[0], key):
                continue
                
            data = getattr(experiments[0], key)
            n_features = data.shape[1]
            matrix_size = self.transform_matrices[key].shape[0]
            
            if not (n_features == matrix_size):
                raise ValueError(f"Matrix dimension {matrix_size} does not match data dimension {n_features} for {key}")
    
    @classmethod
    def from_diagonal(cls, diagonal_elements: Dict[str, torch.Tensor],
                     biases: Optional[Dict[str, torch.Tensor]] = None) -> 'CustomTScaler':
        """
        Create a CustomTScaler from diagonal scaling factors.

        Args:
            diagonal_elements: Dict of 1-D tensors of diagonal entries for each feature.
            biases: Optional dict of bias vectors for each feature.

        Returns:
            CustomTScaler initialised with diagonal transformation matrices.
        """
        transform_matrices = {}
        feature_names = []
        for key, diag in diagonal_elements.items():
            if not (diag.dim() == 1):
                raise ValueError(f"Diagonal elements for {key} must be 1D")
            if not (not torch.any(torch.isclose(diag, torch.tensor(0.0)))):
                raise ValueError(f"Diagonal elements for {key} must be non-zero")
            transform_matrices[key] = torch.diag(diag)
            feature_names.append(key)
        return cls(transform_matrices, biases, feature_names)
    
    def __repr__(self) -> str:
        features = list(self.transform_matrices.keys())
        dims = {k: self.transform_matrices[k].shape[0] for k in features}
        return f"CustomTScaler(features={features}, dimensions={dims})"