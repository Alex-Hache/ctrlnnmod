"""Le maître est-il représentable avec TOUTES les têtes sur ∂S (r_h = ρ) ? Construction exacte."""
import math, sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "memory"))
import m2_boundary as m2
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
RHO, seed = 0.999, 0
u_tr, y_tr, u_te, y_te = [t.double() for t in m2.data(seed)]
A_t, B_t, C_t = [t.double() for t in m2.osc.teacher(seed)]
model = m2.build("psat", seed).double(); m = model.m
sc = float(m2.osc.sim(*m2.osc.teacher(seed), m2.osc.inputs(256, 100, 10 + seed)).std())
with torch.no_grad():
    for p in model.parameters(): p.zero_()
    A = torch.zeros(16, 16); E = torch.zeros(16, 2); gb = torch.zeros(8)
    zmin = m.z_min
    for h, th in enumerate(m2.osc.TH):
        lam = 0.99 * complex(math.cos(math.radians(th)), math.sin(math.radians(th)))
        # |lam - (1-z)| = ρ z  -> racine dans (0, 1)
        f = lambda z: abs(lam - (1 - z)) - RHO * z
        lo, hi = 1e-3, 1.0
        assert f(lo) > 0 > f(hi)
        for _ in range(200):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        z = (lo + hi) / 2
        M = (A_t[2*h:2*h+2, 2*h:2*h+2] - (1 - z) * torch.eye(2)) / z
        A[2*h:2*h+2, 2*h:2*h+2] = M; E[2*h:2*h+2] = B_t[2*h:2*h+2] / z
        q = (z - zmin) / (1 - zmin); gb[h] = math.log(q / (1 - q))
        print(f"θ={th:2d}°  z={z:.4f}  ‖A_hh‖₂={float(torch.linalg.matrix_norm(M, 2)):.6f}")
    for h in range(4, 8):                                   # têtes en trop : r = ρ, état nul, non lues
        A[2*h:2*h+2, 2*h:2*h+2] = RHO * torch.eye(2)
    m.A_.copy_(A); m.E.copy_(E); m.gate_proj.bias.copy_(gb)
    model.C.weight[:, :8] = C_t / sc
    print("géométrie :", m2.geometry(model))
    print("perte train :", float(((model(u_tr) - y_tr) ** 2).mean()), " test :", float(((model(u_te) - y_te) ** 2).mean()))
