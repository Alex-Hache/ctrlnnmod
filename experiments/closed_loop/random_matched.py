"""Paramètres aléatoires à perturbation comparable : PB-REN (γ = 2000, paramètres libres × s)
contre MLP (× 3). Mesure : cible atteinte à 30 s, divergence, écart à la trajectoire du PD.
Usage : python random_matched.py <sortie.json>"""
import json
import statistics as st
import sys

import torch

from arm_boosting import PBREN, evaluate, rollout, sample_x0
from run_arm import T_LONG, make

torch.set_num_threads(2)
x0 = sample_x0(16, gen=torch.Generator().manual_seed(1))


class Zero:
    def reset(self, B):
        pass

    def __call__(self, x, w):
        return torch.zeros(x.shape[0], 2)


with torch.no_grad():
    xb, _ = rollout(Zero(), x0, 150)


def pbren(s, seed):
    torch.manual_seed(seed)
    pol = PBREN(gamma=2000.0)
    with torch.no_grad():
        for n, p in pol.ren.named_parameters():
            if n != "X":
                p.mul_(s)
    return pol


def mlp(seed):
    torch.manual_seed(seed)
    return make("mlp", 3.0)


res = {}
for name, fn in (("pb_ren_g2000_x3", lambda d: pbren(3.0, d)), ("pb_ren_g2000_x10", lambda d: pbren(10.0, d)),
                 ("mlp_x3", mlp)):
    st_reach, st_div, devs, taus = [], [], [], []
    for d in range(100):
        pol = fn(1000 + d)
        e = evaluate(pol, x0, T_LONG)
        st_reach.append(e["reached"])
        st_div.append(e["diverged"])
        with torch.no_grad():
            xs, us = rollout(fn(1000 + d), x0, 150)
        devs.append((xs[..., :2] - xb[..., :2]).norm(dim=-1).amax().item())
        taus.append(us.abs().amax().item())
    res[name] = dict(reached=sum(st_reach) / 100, all_reached=sum(r == 1.0 for r in st_reach) / 100,
                     diverged=sum(st_div) / 100, dev_median=st.median(devs), tau_median=st.median(taus))
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
