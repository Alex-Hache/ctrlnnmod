"""Évaluation sous erreur de modèle (charge utile), relative au contrôleur de base.

Avec une charge plus lourde, la compensation de gravité nominale laisse une erreur statique
même sans politique apprise. On compare donc chaque politique au contrôleur de base seul :
- err_ratio : erreur finale / erreur finale du contrôleur de base (1 = aussi bien que la base)
- settled   : vitesse finale < 1e-2 (la boucle s'est stabilisée sur un équilibre)
- collided  : contact avec l'obstacle
Usage : python eval_mismatch.py <dossier des résultats> <fichier json de sortie>
"""
import glob
import json
import os
import sys

import torch

from arm_boosting import NOMINAL, QSTAR, obstacle_dist, rollout, sample_x0
from run_arm import make

torch.set_num_threads(2)
src, out = sys.argv[1], sys.argv[2]
T = 1500


class Zero:
    def reset(self, B):
        pass

    def __call__(self, x, w):
        return torch.zeros(x.shape[0], 2)


@torch.no_grad()
def stats(pol, x0, plant):
    xs, _ = rollout(pol, x0, T, plant)
    err = (xs[:, -1, :2] - QSTAR).norm(dim=-1)
    return dict(err=err, settled=xs[:, -1, 2:].norm(dim=-1) < 1e-2,
                collided=obstacle_dist(xs[..., :2]).amin(1) < 0,
                bounded=torch.isfinite(xs).all() & (xs.abs().amax() < 1e3))


x0 = sample_x0(100, gen=torch.Generator().manual_seed(4242))
res = {}
for k in (1.0, 1.5, 2.0, 3.0, 5.0):
    plant = dict(NOMINAL, m2=NOMINAL["m2"] * k)
    base = stats(Zero(), x0, plant)
    res[f"base/x{k}"] = dict(err_med=float(base["err"].median()))
    for f in sorted(glob.glob(f"{src}/**/*.pt", recursive=True)):
        name = os.path.basename(f)[:-3]
        kind = name.split("_g")[0]
        gamma = float(name.split("_g")[1].split("_s")[0])
        pol = make(kind, gamma=gamma)
        pol.load_state_dict(torch.load(f))
        s = stats(pol, x0, plant)
        res[f"{name}/x{k}"] = dict(err_ratio_med=float((s["err"] / base["err"].clamp_min(1e-3)).median()),
                                   err_med=float(s["err"].median()),
                                   settled=float(s["settled"].float().mean()),
                                   collided=float(s["collided"].float().mean()),
                                   bounded=bool(s["bounded"]))
        print(name, k, {a: round(b, 3) if isinstance(b, float) else b for a, b in res[f"{name}/x{k}"].items()},
              flush=True)
    json.dump(res, open(out, "w"), indent=1)
