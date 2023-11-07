import torch
import torch.linalg as la
import numpy as np
from torchvision.transforms import Normalize
from autoattack import AutoAttack
from nssmid.ss_models import *
from nssmid.preprocessing import *
from nssmid.losses import *

def evaluate_toy(config):
    seed_everything(config.seed)
    model = getModel(config) 
    xshape = (config.lip_batch_size, config.in_channels)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape))
    model(x)
    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)
    n_layers = len(model.model)
    layer_lip = np.zeros(n_layers)
    layer_norm= np.zeros(n_layers)
    print(f"{config.train_dir}/model.ckpt")
    if model.layer == 'Sandwich':
        psi0 = torch.tensor([-np.log(2)],dtype=torch.float32).to(x.device)
        At0  = torch.eye(1,dtype=torch.float32).to(x.device)
    for k in range(n_layers):
        g = empirical_lipschitz(model.model[k], x)
        layer_lip[k] = g
        x = torch.randn_like(model.model[k](x))
        if model.layer == 'Sandwich':
            if hasattr(model.model[k], 'psi'):
                psi=model.model[k].psi
            else:
                psi=torch.tensor([np.log(2)],dtype=torch.float32).to(x.device)
            f=psi.shape[0]
            Q=model.model[k].Q
            At, B = Q[:,:f].T, Q[:,f:]
            W = 2*torch.exp(-psi).diag() @ B @ At0 @ torch.exp(psi0).diag()
            At0, psi0 = At, psi
        elif model.layer == 'Orthogon':
            W = model.model[k].Q
        elif model.layer == 'Aol':
            Weight = model.model[k].weights
            T = 1/torch.sqrt(torch.abs(Weight.T @ Weight).sum(1))
            W = model.model[k].scale * Weight * T

        _, S, _ = la.svd(W)
        w = S[0].item()
        layer_norm[k] = w
        print(f"Layer: {k}, Lip: {g:.4f}, Norm: {w:4f}")

    np.savetxt(f"{config.train_dir}/layer_lip.csv", layer_lip)
    np.savetxt(f"{config.train_dir}/layer_norm.csv", layer_norm)

def evaluate_model(config):
    seed_everything(config.seed)
    model = getModel(config) 
    xshape = (config.lip_batch_size, config.in_channels)
    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)
    config.lmi = 'lipschitz'
    criterion = getLoss(config, model)
    return criterion.lmi.L

def empirical_lipschitz(model, x, eps=0.05):

    norms = lambda X: X.view(X.shape[0], -1).norm(dim=1) ** 2
    gam = 0.0
    for r in range(10):
        dx = torch.zeros_like(x)
        dx.uniform_(-eps,eps)
        x.requires_grad = True
        dx.requires_grad = True
        optimizer = torch.optim.Adam([x, dx], lr=1E-1)
        iter, j = 0, 0
        LipMax = 0.0
        while j < 50:
            LipMax_1 = LipMax
            optimizer.zero_grad()
            dy = model(x + dx) - model(x)
            Lip = norms(dy) / (norms(dx) + 1e-6)
            Obj = -Lip.sum()
            Obj.backward()
            optimizer.step()
            LipMax = Lip.max().item()
            iter += 1
            j += 1
            if j >= 5:
                if LipMax < LipMax_1 + 1E-3:  
                    optimizer.param_groups[0]["lr"] /= 10.0
                    j = 0

                if optimizer.param_groups[0]["lr"] <= 1E-5:
                    break
        
        gam = max(gam, np.sqrt(LipMax))

    return gam 