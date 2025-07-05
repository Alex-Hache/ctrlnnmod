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
                
                # Application de la transformation inverse
                T_inv = torch.inverse(T)
                original_data = torch.matmul(data - b, T_inv.T)
                
                # Préservation du gradient si nécessaire
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
        """Charge les paramètres dans un nouveau scaler"""
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
        """Crée une copie exacte du scaler"""
        params = self.save_parameters()
        return self.load_parameters(params)
    

class MinMaxScaler(BaseScaler):
    """Implémentation de la normalisation min-max"""
    
    def __init__(self, feature_names: List[str] = ['u', 'y', 'x', 'd'],
                 feature_range: tuple = (0, 1)):
        super().__init__(feature_names)
        self.feature_range = feature_range
        self.data_min: Dict[str, torch.Tensor] = {}
        self.data_max: Dict[str, torch.Tensor] = {}
        
    def fit(self, experiments: List['Experiment']) -> None:
        """Calcule les min et max pour chaque feature"""
        for key in self.feature_names:
            if not hasattr(experiments[0], key):
                continue
                
            # Concatenation des données pour toutes les expériences
            all_data = torch.cat([getattr(exp, key) for exp in experiments], dim=0)
            
            # Calcul min/max
            self.data_min[key] = all_data.min(dim=0).values
            self.data_max[key] = all_data.max(dim=0).values
            
            # Calcul de la matrice de transformation
            data_range = self.data_max[key] - self.data_min[key]
            scale = torch.where(data_range == 0,
                              torch.ones_like(data_range),
                              data_range)
            
            # Création de la matrice diagonale de transformation
            n_features = all_data.shape[1]
            T = torch.diag(1.0 / scale)
            b = -self.data_min[key] / scale
            
            self.transform_matrices[key] = T
            self.transform_biases[key] = b
            
        self.is_fitted = True

class StandardScaler(BaseScaler):
    """Implémentation de la standardisation (moyenne=0, variance=1)"""
    
    def __init__(self, feature_names: List[str] = ['u', 'y', 'x', 'd']):
        super().__init__(feature_names)
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}
        
    def fit(self, experiments: List['Experiment']) -> None:
        """Calcule la moyenne et l'écart-type pour chaque feature"""
        for key in self.feature_names:
            if not hasattr(experiments[0], key):
                continue
                
            # Concatenation des données pour toutes les expériences
            all_data = torch.cat([getattr(exp, key) for exp in experiments], dim=0)
            
            # Calcul moyenne/écart-type
            self.means[key] = all_data.mean(dim=0)
            self.stds[key] = all_data.std(dim=0)
            
            # Gestion des variables constantes
            self.stds[key] = torch.where(self.stds[key] == 0,
                                       torch.ones_like(self.stds[key]),
                                       self.stds[key])
            
            # Création de la matrice diagonale de transformation
            n_features = all_data.shape[1]
            T = torch.diag(1.0 / self.stds[key])
            b = -self.means[key] / self.stds[key]
            
            self.transform_matrices[key] = T
            self.transform_biases[key] = b
            
        self.is_fitted = True

class CustomTScaler(BaseScaler):
    """
    Scaler permettant de définir directement les matrices de transformation linéaire.
    Pour chaque feature (u, y, x), la transformation est de la forme:
        x_transformed = x @ T.T + b
    où T est une matrice carrée inversible et b un vecteur de biais.
    """
    
    def __init__(self, transform_matrices: Dict[str, torch.Tensor],
                 transform_biases: Optional[Dict[str, torch.Tensor]] = None,
                 feature_names: List[str] = ['u', 'y', 'x', 'd']):
        """
        Args:
            transform_matrices: Dictionnaire contenant les matrices T pour chaque feature
            transform_biases: Dictionnaire contenant les vecteurs de biais pour chaque feature
                            Si None, les biais sont initialisés à zéro
            feature_names: Liste des noms des features à transformer
        """
        super().__init__(feature_names)
        
        # Vérification des features fournies
        for key in transform_matrices.keys():
            assert key in feature_names, f"Feature {key} not in feature_names {feature_names}"
        
        # Vérification des matrices de transformation
        for key, matrix in transform_matrices.items():
            # Vérification que c'est bien un tenseur
            assert isinstance(matrix, torch.Tensor), f"Matrix for {key} must be a torch.Tensor"
            
            # Vérification que la matrice est 2D
            assert matrix.dim() == 2, f"Matrix for {key} must be 2D, got {matrix.dim()}D"
            
            # Vérification que la matrice est carrée
            n, m = matrix.shape
            assert n == m, f"Matrix for {key} must be square, got shape {matrix.shape}"
            
            # Vérification que la matrice est inversible
            det = torch.linalg.det(matrix)
            assert not torch.isclose(det, torch.tensor(0.0)), \
                   f"Matrix for {key} must be invertible, got determinant {det}"
        
        self.transform_matrices = transform_matrices
        
        # Initialisation ou vérification des biais
        if transform_biases is None:
            self.transform_biases = {
                key: torch.zeros(matrix.shape[0])
                for key, matrix in transform_matrices.items()
            }
        else:
            # Vérification des biais fournis
            for key, bias in transform_biases.items():
                assert key in transform_matrices, f"Bias provided for {key} but no matrix found"
                assert isinstance(bias, torch.Tensor), f"Bias for {key} must be a torch.Tensor"
                assert bias.dim() == 1, f"Bias for {key} must be 1D, got {bias.dim()}D"
                assert bias.shape[0] == transform_matrices[key].shape[0], \
                       f"Bias dimension {bias.shape[0]} does not match matrix dimension {transform_matrices[key].shape[0]}"
            self.transform_biases = transform_biases
        
        self.is_fitted = True  # Le scaler est déjà configuré avec les matrices fournies
    
    def fit(self, experiments: List['Experiment']) -> None:
        """
        Cette méthode ne fait rien car les matrices sont déjà définies,
        mais on vérifie la cohérence des dimensions avec les données.
        """
        for key in self.transform_matrices.keys():
            if not hasattr(experiments[0], key):
                continue
                
            data = getattr(experiments[0], key)
            n_features = data.shape[1]
            matrix_size = self.transform_matrices[key].shape[0]
            
            assert n_features == matrix_size, \
                   f"Matrix dimension {matrix_size} does not match data dimension {n_features} for {key}"
    
    @classmethod
    def from_diagonal(cls, diagonal_elements: Dict[str, torch.Tensor],
                     biases: Optional[Dict[str, torch.Tensor]] = None) -> 'CustomTScaler':
        """
        Crée un CustomTScaler à partir d'éléments diagonaux.
        
        Args:
            diagonal_elements: Dictionnaire contenant les éléments diagonaux pour chaque feature
            biases: Dictionnaire optionnel contenant les biais pour chaque feature
        
        Returns:
            CustomTScaler initialisé avec des matrices diagonales
        """
        transform_matrices = {}
        feature_names = []
        for key, diag in diagonal_elements.items():
            assert diag.dim() == 1, f"Diagonal elements for {key} must be 1D"
            assert not torch.any(torch.isclose(diag, torch.tensor(0.0))), \
                   f"Diagonal elements for {key} must be non-zero"
            transform_matrices[key] = torch.diag(diag)
            feature_names.append(key)
        return cls(transform_matrices, biases, feature_names)
    
    def __repr__(self) -> str:
        features = list(self.transform_matrices.keys())
        dims = {k: self.transform_matrices[k].shape[0] for k in features}
        return f"CustomTScaler(features={features}, dimensions={dims})"