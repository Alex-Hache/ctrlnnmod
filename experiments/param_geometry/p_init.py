"""H∞ (sqrtm, Riccati) : κ(JJᵀ) avec l'initialisation aléatoire de P contre P = I (nx = 6, 8 graines)."""
import sys
import numpy as np
import torch
sys.argv = ["x"]
exec(open("experiments/param_geometry/lmi_fingerprint.py").read().split("res = []")[0])
nx = 6
rows = ["| Paramétrisation | PD | P au départ | κ(JJᵀ) médian | pire κ | max Re λ(A) médian | λ_min(P) médian | rang min / dim |",
        "|---|---|---|---|---|---|---|---|"]
for kind in ("hinf_sqrtm", "hinf_riccati"):
    for pd in ("geotorch", "bm"):
        for init in ("aléatoire", "I"):
            K, RE, LM, RK = [], [], [], []
            for seed in range(8):
                torch.manual_seed(seed)
                m = build(kind, nx, 2, 2, pd)
                if init == "I":
                    with torch.no_grad():
                        if pd == "geotorch":
                            m.P = torch.eye(nx)
                        else:
                            m.parametrizations.P.original.copy_(torch.eye(nx))
                fp = fingerprint(m); K.append(fp["kappa_range"]); RK.append(fp["rank"]); D = fp["dim"]
                RE.append(float(torch.linalg.eigvals(m._frame()[0].detach()).real.max()))
                LM.append(float(torch.linalg.eigvalsh(m.P.detach())[0]))
            rows.append(f"| {kind} | {pd} | {init} | {np.median(K):.1e} | {max(K):.1e} | {np.median(RE):.3f} | {np.median(LM):.1e} | {min(RK)} / {D} |")
txt = "\n".join(rows)
print(txt)
open("experiments/param_geometry/analysis/p_init.md", "w").write(txt + "\n")
