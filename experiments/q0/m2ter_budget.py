"""Q0 — préalable au bras « lr par bloc » : répartition du budget r_h = (part Ã) + (part B̃·C) à W₀',
et sens de variation de chaque part le long de −∇L (flot GD) et du premier pas d'Adam (−signe ∇L).
Le maître (m2_teacher_on_boundary.py) a une part B̃·C nulle et une part Ã ≈ ρ sur 4 têtes."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m2_boundary as m2
import m2bis as mb
torch.set_num_threads(1)


def parts(m, A, B, C):
    H, k, nq = m.H, m.k, m.nq
    a = torch.linalg.matrix_norm(A.reshape(H, k, H, k).transpose(1, 2), ord=2).sum(1)
    c = C.reshape(nq, H, k).norm(dim=2).sum(1)
    b = B.reshape(H, k, nq).norm(dim=1) @ c
    return a, b


for seed in range(5):
    u, y, _, _ = m2.data(seed)
    mod = mb.build("pint", seed); m = mod.m
    ((mod(u) - y) ** 2).mean().backward()
    P = dict(m.named_parameters())
    th = {n: P[n].detach().clone() for n in mb.RP}
    gr = {n: P[n].grad.detach().clone() for n in mb.RP}
    gn = torch.sqrt(sum((gr[n] ** 2).sum() for n in mb.RP))
    with torch.no_grad():
        a0, b0 = parts(m, th["A_"], th["B_"], th["C"])
        eps = 1e-4                                              # pas GD normalisé, ordre 1
        a1, b1 = parts(m, *(th[n] - eps * gr[n] / gn for n in mb.RP))
        lr = 1e-4                                               # premier pas d'Adam
        a2, b2 = parts(m, *(th[n] - lr * torch.sign(gr[n]) for n in mb.RP))
    print(f"graine {seed} : r = {float((a0 + b0).max()):.3f} | part Ã moy. {float(a0.mean()):.3f} "
          f"[{float(a0.min()):.3f}, {float(a0.max()):.3f}] | part B̃·C moy. {float(b0.mean()):.3f} "
          f"[{float(b0.min()):.3f}, {float(b0.max()):.3f}] | |B̃|/|Ã| = {float(th['B_'].abs().mean() / th['A_'].abs().mean()):.3f}")
    print(f"   GD (par unité de pas) : d(part Ã) moy. {float((a1 - a0).mean() / eps):+.3f}, d(part B̃·C) moy. {float((b1 - b0).mean() / eps):+.3f}"
          f" | têtes où B̃·C baisse : {int(((b1 - b0) < 0).sum())}/8")
    print(f"   Adam lr={lr:g} (1 pas)  : Δ(part Ã) moy. {float((a2 - a0).mean()):+.2e}, Δ(part B̃·C) moy. {float((b2 - b0).mean()):+.2e}"
          f" | têtes où B̃·C baisse : {int(((b2 - b0) < 0).sum())}/8")
