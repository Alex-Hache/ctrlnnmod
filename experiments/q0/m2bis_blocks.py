"""Q0 M2-bis — quel bloc (Ã, B̃, C) porte P-int jusqu'à ∂S ? (graine 0, après la calibration)

Pour lr ∈ {1e-3, 1e-4} : Adam jusqu'au contact t_c, puis r_max de θ₀' avec un seul bloc déplacé à θ_tc
(déplacement réel, et 20 orientations aléatoires de même |Δ| par coordonnée). Échelle : |Δ| moyen / |θ₀'| moyen.
Teste le mécanisme « B̃ cause le contact », préalable à un lr par bloc.
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m2_boundary as m2
import m2bis as mb
torch.set_num_threads(1)
u, y, _, _ = m2.data(0)
for lr in (1e-3, 1e-4):
    mod = mb.build("pint", 0); m = mod.m; P = dict(m.named_parameters())
    th0 = {n: P[n].detach().clone() for n in mb.RP}
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    for it in range(1000):
        with torch.no_grad():
            r = mb.r_raw(m.H, m.k, m.nq, *(P[n] for n in mb.RP))
        if float(r.max()) >= m2.RHO - m2.TOL:
            break
        opt.zero_grad(); ((mod(u) - y) ** 2).mean().backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 10.0); opt.step()
    th = {n: P[n].detach().clone() for n in mb.RP}
    g = torch.Generator().manual_seed(0)
    print(f"lr={lr:g} : t_c={it}, r_max={float(r.max()):.3f} (départ 0,899)")
    for X in mb.RP:
        D = th[X] - th0[X]
        real = {n: (th[n] if n == X else th0[n]) for n in mb.RP}
        with torch.no_grad():
            rr = float(mb.r_raw(m.H, m.k, m.nq, *(real[n] for n in mb.RP)).max())
            sg = (2 * torch.randint(0, 2, (20, *D.shape), generator=g) - 1).float()
            mir = {n: (th0[n] + sg * D if n == X else th0[n].expand(20, *th0[n].shape)) for n in mb.RP}
            rm = mb.r_raw(m.H, m.k, m.nq, *(mir[n] for n in mb.RP)).max(-1).values
        print(f"  {X:2s} seul : r_max réel={rr:.3f} | orientations aléatoires méd.={float(rm.median()):.3f}"
              f" [{float(rm.min()):.3f}, {float(rm.max()):.3f}] | |Δ|/|θ₀'| = {float(D.abs().mean() / th0[X].abs().mean()):.2f}")
