"""Q0 M2 — pourquoi P-int touche ∂S au premier pas ? Décomposition de r_h et d'un pas d'Adam (graine 0)."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m2_boundary as m2
torch.set_num_threads(1)

def parts(m):
    H, k = m.H, m.k
    a = torch.linalg.matrix_norm(m.A_.view(H, k, H, k).permute(0, 2, 1, 3), ord=2).sum(1)
    c = m.C.view(m.nq, H, k).norm(dim=2).sum(1)
    b = m.B_.view(H, k, m.nq).norm(dim=1) @ c
    return a, b, c

u, y, _, _ = m2.data(0)
for arm in ("psat", "pint"):
    mod = m2.build(arm, 0); m = mod.m
    with torch.no_grad():
        a, b, c = parts(m)
    print(f"[{arm}] r_h max = {float((a + b).max()):.3f} | part A = {float(a.mean()):.3f} | part B·C = {float(b.mean()):.3f}"
          f" | c_q moyen = {float(c.mean()):.2f} | |Ã| moyen = {float(m.A_.detach().abs().mean()):.4f} | |B̃| moyen = {float(m.B_.detach().abs().mean()):.4f}")

# Un pas d'Adam (lr 1e-2) depuis P-int : au premier pas, Δθ = −lr·signe(g) exactement.
mod = m2.build("pint", 0); m = mod.m
loss = ((mod(u) - y) ** 2).mean(); loss.backward()
torch.nn.utils.clip_grad_norm_(mod.parameters(), 10.0)
with torch.no_grad():
    a0, b0, _ = parts(m); r0 = a0 + b0
    saved = {n: p.detach().clone() for n, p in m.named_parameters()}
    def r_after(step, names):
        for n, p in m.named_parameters():
            p.copy_(saved[n] + (step[n] if n in names else 0))
        a, b, _ = parts(m); out = a + b
        for n, p in m.named_parameters(): p.copy_(saved[n])
        return out
    lr = 1e-2
    adam = {n: -lr * torch.sign(p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in m.named_parameters()}
    print(f"\n[pint] r_h max avant = {float(r0.max()):.3f} (ρ = 0,999)")
    for lab, names in (("Ã seul", {"A_"}), ("B̃ seul", {"B_"}), ("C seul", {"C"}), ("Ã, B̃, C", {"A_", "B_", "C"})):
        print(f"  pas d'Adam, {lab:9s}: r_h max après = {float(r_after(adam, names).max()):.3f}")
    # même taille, signes aléatoires : le gradient n'est pas nécessaire pour gonfler r
    g = torch.Generator().manual_seed(0); rs = []
    for _ in range(20):
        rnd = {n: lr * (2 * torch.randint(0, 2, p.shape, generator=g) - 1).float() for n, p in m.named_parameters()}
        rs.append(float(r_after(rnd, {"A_", "B_", "C"}).max()))
    print(f"  pas de signes ALÉATOIRES, même taille : r_h max après = {min(rs):.3f} à {max(rs):.3f} (20 tirages)")
    # pas de gradient pur, très petit : ordre 1 seulement (direction réelle du gradient)
    for eta in (1e-4, 1e-3):
        gd = {n: -eta * (p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in m.named_parameters()}
        d = r_after(gd, {"A_", "B_", "C"}) - r0
        print(f"  GD η={eta:g} : Δr_h max = {float(d.max()):+.2e}, Δr_h min = {float(d.min()):+.2e}")
    print()
    for lr2 in (1e-3, 3e-3):
        a2 = {n: -lr2 * torch.sign(p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in m.named_parameters()}
        rs = []
        for _ in range(20):
            rnd = {n: lr2 * (2 * torch.randint(0, 2, p.shape, generator=g) - 1).float() for n, p in m.named_parameters()}
            rs.append(float(r_after(rnd, {"A_", "B_", "C"}).max()))
        print(f"  lr={lr2:g} : pas d'Adam → r_h max = {float(r_after(a2, {'A_', 'B_', 'C'}).max()):.3f} ; signes aléatoires → {min(rs):.3f} à {max(rs):.3f}")
