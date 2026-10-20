"""Riccati H∞ : κ(JJᵀ) et marge de stabilité selon nu + ny à nx fixé.
Hypothèse : la dissipation BBᵀ/γ² + GᵀG a un rang ≤ nu + ny ; si nu + ny < nx, elle est singulière."""
import sys
import numpy as np
import torch
sys.argv = ["x"]
exec(open("experiments/param_geometry/lmi_fingerprint.py").read().split("res = []")[0])
nx = 6
rows = ["| nu = ny | nu + ny | rang dissipation | κ(JJᵀ) médian | max Re λ(A) médian |", "|---|---|---|---|---|"]
for k in (1, 2, 3, 4, 6):
    K, RE, RK = [], [], []
    for seed in range(8):
        torch.manual_seed(seed)
        m = build("hinf_riccati", nx, k, k, "geotorch")
        K.append(fingerprint(m)["kappa_range"])
        A = m._frame()[0].detach()
        RE.append(float(torch.linalg.eigvals(A).real.max()))
        D = (m.B @ m.B.T / m.gamma ** 2 + m.G.T @ m.G).detach()
        RK.append(int(torch.linalg.matrix_rank(D)))
    rows.append(f"| {k} | {2 * k} | {min(RK)} / {nx} | {np.median(K):.1e} | {np.median(RE):.4f} |")
txt = "\n".join(rows)
print(txt)
open("experiments/param_geometry/analysis/riccati_rank.md", "w").write(txt + "\n")
