import pytest
import torch
from ctrlnmod.models.ssmodels.continuous.linearization import FLNSSM
from ctrlnmod.models.ssmodels.continuous.linear import (
    L2BoundedLinear,
    ExoH2Linear,
    ExoL2BoundedLinear,
    ExoSSLinear,
    SSLinear,
    H2Linear
)
from ctrlnmod.integrators import RK4Simulator

# 📌 Paramètres de dimensions pour les modèles
@pytest.fixture
def dims():
    return {
        'nu': 2,
        'ny': 3,
        'nx': 4,
        'nd': 1
    }

# 📌 Classes de modèles linéaires avec leurs paramètres
@pytest.fixture
def linear_model_classes():
    return [
        L2BoundedLinear,
        ExoH2Linear,
        ExoL2BoundedLinear,
        ExoSSLinear,
        SSLinear,
        H2Linear
    ]

# 📌 Types de contrôleurs
@pytest.fixture
def controller_types():
    return ["output_feedback", "state_feedback", "beta_output_feedback", "beta_state_feedback"]

# 🧪 Test paramétrisé avec produit cartésien
@pytest.mark.parametrize("linear_model_class", [
    L2BoundedLinear,
    ExoH2Linear,
    ExoL2BoundedLinear,
    ExoSSLinear,
    SSLinear,
    H2Linear
])
@pytest.mark.parametrize("controller_type", [
    "output_feedback", 
    "state_feedback", 
    "beta_output_feedback", 
    "beta_state_feedback"
])
def test_build_flnssm(dims, controller_type, linear_model_class):
    nu, ny, nx, nd = dims['nu'], dims['ny'], dims['nx'], dims['nd']
    
    # ⚠️ Gestion des constructeurs différents
    if "Exo" in linear_model_class.__name__:
        if linear_model_class == ExoH2Linear:
            linear_model = linear_model_class(nu, ny, nx, nd, 1.5)
        elif linear_model_class == ExoL2BoundedLinear:
            linear_model = linear_model_class(nu, ny, nx, nd, 2.0)
        else:  # ExoSSLinear
            linear_model = linear_model_class(nu, ny, nx, nd)
    else:
        if linear_model_class == L2BoundedLinear:
            linear_model = linear_model_class(nu, ny, nx, 2.0)
        elif linear_model_class == SSLinear:
            linear_model = linear_model_class(nu, ny, nx, 1.5)
        else:  # H2Linear
            linear_model = linear_model_class(nu, ny, nx, 2.0)
        nd = None
    # ✅ On instancie le modèle non linéaire
    model = FLNSSM(nu, ny, nx, linear_model, controller_type, nd)
    
    # 🧪 On vérifie simplement que l'objet est instancié
    assert isinstance(model, FLNSSM)

    # Simulating with a very small step size in order not to get instaiblity
    sim_model = RK4Simulator(model, ts=1e-4)  

    nb, n_seq = 10, 100
    u = torch.randn((nb, n_seq, nu))
    x0 = torch.zeros((nb, nx))
    if "Exo" in linear_model_class.__name__:
        d = torch.zeros((nb, n_seq, nd))
    else:
        d = None


    with torch.no_grad():
        x_sim, y_sim = sim_model(u, x0, d)

    assert isinstance(x_sim, torch.Tensor)
    assert isinstance(y_sim, torch.Tensor)

    assert x_sim.shape == (10, 100, nx)
    assert y_sim.shape == (10, 100, ny)
