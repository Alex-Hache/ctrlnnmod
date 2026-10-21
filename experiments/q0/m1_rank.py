"""
Q0 — mesure M1 : rang et conditionnement de J = DΦ sur les blocs de 2 (oscillateurs).

Question : la paramétrisation Φ : θ → W des blocs perd-elle du rang, ou a-t-elle un κ(JJᵀ) élevé,
le long d'un run échoué ? Le REN sert de témoin (il réussit sur la même tâche).

W = coordonnées du modèle libre (mêmes équations, sans contrainte). Φ ne modifie que (A, B) pour
les blocs. Pour chaque instantané θ_t (t = 0, 250, …, 2000), on mesure :
  rang de J (σ > 1e-6 σ_max), déficit = dim W − rang, κ(JJᵀ) sur l'image, nombre de têtes
  saturées (r_h brut > ρ, variante proj), marge du certificat en W, part de g = ∇_W L hors de
  l'image de J, part de g dans les directions lentes (σ² < 1e-3 σ²_max), ‖Jᵀg‖ / (σ_max ‖g‖).
On mesure aussi J en θ* (préimage exacte du maître) quand elle existe.

Contrôles : L(Φ(θ)) = perte du module ; J par différences finies ; L(θ*) ≈ 0.
Usage : python experiments/q0/m1_rank.py   (≈ 5 min sur 2 cœurs)
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import torch
from torch.func import functional_call, jacrev

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "memory"))
import oscillator as osc  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m1"))
KINDS = {"blockp_nogate": 3e-3, "blockp_half": 3e-3, "blockp": 1e-2, "block": 1e-2, "blockh": 3e-3, "ren": 3e-3}
ITERS, EVERY, SEED = int(os.environ.get("ITERS", 2000)), int(os.environ.get("EVERY", 250)), 0
RTOL, SLOW = 1e-6, 1e-3


# ------------------------------------------------------------------------------------------
# Φ : θ → W et perte en W (modèle libre), pour les blocs et pour le REN
# ------------------------------------------------------------------------------------------
def flat(d, keys):
    return torch.cat([d[k].reshape(-1) for k in keys])


def unflat(v, keys, shapes):
    out, i = {}, 0
    for k in keys:
        n = math.prod(shapes[k])
        out[k] = v[i:i + n].reshape(shapes[k])
        i += n
    return out


class BlockMaps:
    WKEYS = ("A", "B", "C", "D", "E", "b", "bx", "gW", "gb", "Cw", "Cb")

    def __init__(self, model):
        self.mod, m = model, model.m
        self.H, self.k, self.nq, self.rho, self.norm = m.H, m.k, m.nq, m.rho, m.norm
        self.z_min, self.ung = m.z_min, m.ungated
        self.tkeys = [k for k, _ in model.named_parameters()]
        self.tshapes = {k: tuple(p.shape) for k, p in model.named_parameters()}
        self.wshapes = dict(A=(m.nx, m.nx), B=(m.nx, m.nq), C=(m.nq, m.nx), D=tuple(m.D.shape),
                            E=tuple(m.E.shape), b=(m.nq,), bx=(m.nx,), gW=tuple(m.gate_proj.weight.shape),
                            gb=(m.H,), Cw=tuple(model.C.weight.shape), Cb=tuple(model.C.bias.shape))

    def rows(self, A, B, C):
        H, k = self.H, self.k
        Ab = A.view(H, k, H, k).permute(0, 2, 1, 3)
        nA = torch.linalg.matrix_norm(Ab, ord=2)
        nB = B.view(H, k, self.nq).norm(dim=1)
        c = C.view(self.nq, H, k).norm(dim=2).sum(1)
        return nA.sum(1) + nB @ c

    def scale(self, r, c):
        return osc_row_scale(r, self.rho, self.norm, c)

    def theta(self):
        return flat(dict(self.mod.named_parameters()), self.tkeys).detach()

    def phi(self, th):
        p = unflat(th, self.tkeys, self.tshapes)
        r = self.rows(p["m.A_"], p["m.B_"], p["m.C"])
        s = self.scale(r, p["m.c"]).repeat_interleave(self.k)[:, None]
        W = dict(A=p["m.A_"] * s, B=p["m.B_"] * s, C=p["m.C"], D=p["m.D"], E=p["m.E"], b=p["m.b"], bx=p["m.bx"],
                 gW=p["m.gate_proj.weight"], gb=p["m.gate_proj.bias"], Cw=p["C.weight"], Cb=p["C.bias"])
        return flat(W, self.WKEYS)

    def loss_W(self, w, u, y):
        W = unflat(w, self.WKEYS, self.wshapes)
        z = self.z_min + (1 - self.z_min) * torch.sigmoid(u @ W["gW"].T + W["gb"])
        if self.ung:
            z = torch.cat([torch.ones_like(z[..., :self.ung]), z[..., self.ung:]], -1)
        z = z.repeat_interleave(self.k, dim=-1)
        x, xs = u.new_zeros(u.shape[0], W["A"].shape[0]), []
        for t in range(u.shape[1]):
            ut = u[:, t]
            f = x @ W["A"].T + torch.tanh(x @ W["C"].T + ut @ W["D"].T + W["b"]) @ W["B"].T + ut @ W["E"].T + W["bx"]
            x = (1 - z[:, t]) * x + z[:, t] * f
            xs.append(x)
        yh = torch.stack(xs, 1) @ W["Cw"].T + W["Cb"]
        return ((yh - y) ** 2).mean()

    def extra(self, th, w):
        """Têtes saturées (proj) et marge du certificat en W."""
        p, W = unflat(th, self.tkeys, self.tshapes), unflat(w, self.WKEYS, self.wshapes)
        r_raw = self.rows(p["m.A_"], p["m.B_"], p["m.C"])
        r_W = self.rows(W["A"], W["B"], W["C"])
        bound = 1.0 if self.norm == "homo" else self.rho
        return dict(n_sat=int((r_raw > self.rho).sum()) if self.norm == "proj" else None,
                    margin=float(bound - r_W.max()), rowsum_max=float(r_W.max()))


def osc_row_scale(r, rho, norm, c):
    from ctrlnmod.models.discrete.linf_gated import _row_scale
    return _row_scale(r, rho, norm, c)


class RenMaps:
    WKEYS = ("A", "Bw", "Bu", "C1", "bx", "loglam", "D12", "C2", "D21", "D22", "bv", "by")

    def __init__(self, model):
        self.mod, m = model, model.m
        self.nx, self.nq, self.eps, self.ab = m.nx, m.nq, m.eps, m.alpha_bar
        self.tkeys = [k for k, _ in model.named_parameters()]
        self.tshapes = {k: tuple(p.shape) for k, p in model.named_parameters()}
        nx, nq, nu, ny = m.nx, m.nq, m.nu, m.ny
        self.wshapes = dict(A=(nx, nx), Bw=(nx, nq), Bu=(nx, nu), C1=(nq, nx), bx=(nx,), loglam=(nq,),
                            D12=(nq, nu), C2=(ny, nx), D21=(ny, nq), D22=(ny, nu), bv=(nq,), by=(ny,))

    def theta(self):
        return flat(dict(self.mod.named_parameters()), self.tkeys).detach()

    def phi(self, th):
        p = unflat(th, self.tkeys, self.tshapes)
        nx, nq = self.nx, self.nq
        X = p["m.X"]
        H = X.T @ X + self.eps * torch.eye(X.shape[0], dtype=X.dtype)
        H11, H21, H22 = H[:nx, :nx], H[nx:nx + nq, :nx], H[nx:nx + nq, nx:nx + nq]
        H31, H32, P = H[nx + nq:, :nx], H[nx + nq:, nx:nx + nq], H[nx + nq:, nx + nq:]
        E = 0.5 * (H11 + P / self.ab ** 2 + p["m.Y"] - p["m.Y"].T)
        lam = 0.5 * H22.abs().sum(1)
        Ei = torch.linalg.inv(E)
        W = dict(A=Ei @ H31, Bw=Ei @ H32, Bu=Ei @ p["m.B2"], C1=-H21, bx=Ei @ p["m.bx"], loglam=lam.log(),
                 D12=p["m.D12"], C2=p["m.C2"], D21=p["m.D21"], D22=p["m.D22"], bv=p["m.bv"], by=p["m.by"])
        return flat(W, self.WKEYS)

    def loss_W(self, w, u, y):
        W = unflat(w, self.WKEYS, self.wshapes)
        lam = W["loglam"].exp()
        x, ys = u.new_zeros(u.shape[0], self.nx), []
        for t in range(u.shape[1]):
            ut = u[:, t]
            v = torch.tanh((x @ W["C1"].T + ut @ W["D12"].T + W["bv"]) / lam)
            ys.append(x @ W["C2"].T + v @ W["D21"].T + ut @ W["D22"].T + W["by"])
            x = x @ W["A"].T + v @ W["Bw"].T + ut @ W["Bu"].T + W["bx"]
        return ((torch.stack(ys, 1) - y) ** 2).mean()

    def extra(self, th, w):
        return dict(n_sat=None, margin=None, rowsum_max=None)


# ------------------------------------------------------------------------------------------
# Diagnostics en un point
# ------------------------------------------------------------------------------------------
def diagnose(maps, th, u, y):
    th = th.double()
    w = maps.phi(th)
    J = jacrev(maps.phi)(th)                                     # (dim W, dim θ)
    U, S, _ = torch.linalg.svd(J, full_matrices=False)
    dimW = J.shape[0]
    S = S[:dimW]
    U = U[:, :dimW]
    rank = int((S > RTOL * S[0]).sum())
    wr = w.detach().requires_grad_(True)
    L = maps.loss_W(wr, u, y)
    (g,) = torch.autograd.grad(L, wr)
    gn2 = float(g @ g)
    coef = U[:, :rank].T @ g
    frac_out = max(0.0, 1 - float(coef @ coef) / gn2) if gn2 > 0 else float("nan")
    slow = S[:rank] ** 2 < SLOW * S[0] ** 2
    frac_slow = float((coef[slow] ** 2).sum()) / gn2 if gn2 > 0 else float("nan")
    Jtg = J.T @ g
    out = dict(loss=float(L), dimW=dimW, dimTheta=J.shape[1], rank=rank, deficit=dimW - rank,
               sigma_max=float(S[0]), sigma_min=float(S[-1]), sigma_min_img=float(S[rank - 1]),
               kappa_img=float((S[0] / S[rank - 1]) ** 2), n_slow=int(slow.sum()),
               frac_out=frac_out, frac_slow=frac_slow,
               jtg_ratio=float(Jtg.norm() / (S[0] * g.norm())) if gn2 > 0 else float("nan"),
               g_norm=math.sqrt(gn2))
    out.update(maps.extra(th, w.detach()))
    return out


def fd_check(maps, th):
    """Écart relatif entre J·v et la différence finie centrée de Φ."""
    th = th.double()
    v = torch.randn_like(th)
    v /= v.norm()
    h = 1e-6
    fd = (maps.phi(th + h * v) - maps.phi(th - h * v)) / (2 * h)
    Jv = torch.func.jvp(maps.phi, (th,), (v,))[1]
    return float((fd - Jv).norm() / Jv.norm().clamp(min=1e-30))


# ------------------------------------------------------------------------------------------
# Préimage du maître (blocs) : heads 0..3 = oscillateurs, heads 4..7 = 0
# ------------------------------------------------------------------------------------------
def teacher_theta(maps, kind, sc):
    m = maps.mod.m
    if m.norm == "sat":
        return None, "aucune : somme de lignes < ρ = 0,99 ; le maître (≥ 0,99) est hors de S, dans l'adhérence"
    A_t, B_t, C_t = osc.teacher(SEED)
    A_t, B_t, C_t = A_t.double(), B_t.double(), C_t.double()
    nx, H, k, zmin = m.nx, m.H, m.k, m.z_min
    p = {kk: torch.zeros(s, dtype=torch.float64) for kk, s in maps.tshapes.items()}
    zstar = 0.99
    gb = math.log((zstar - zmin) / (1 - zmin)) - math.log(1 - (zstar - zmin) / (1 - zmin))
    p["m.gate_proj.bias"][:] = gb
    A = torch.zeros(nx, nx, dtype=torch.float64)
    E = torch.zeros(nx, B_t.shape[1], dtype=torch.float64)
    for h in range(4):
        z = 1.0 if h < m.ungated else zstar
        sl = slice(2 * h, 2 * h + 2)
        A[sl, sl] = (A_t[sl, sl] - (1 - z) * torch.eye(2, dtype=torch.float64)) / z
        E[sl] = B_t[sl] / z
    r = torch.linalg.matrix_norm(A.view(H, k, H, k).permute(0, 2, 1, 3), ord=2).sum(1)
    if m.norm == "proj":
        if float(r.max()) >= m.rho:
            return None, f"aucune : r_max = {float(r.max()):.4f} ≥ ρ"
        p["m.A_"] = A
    else:                                                        # homo : W = A_/(r + softplus(c)), A_ = A
        sp = (1 - r).clamp(min=1e-12)
        sp[r == 0] = 1.0
        p["m.A_"] = A
        p["m.c"] = torch.log(torch.expm1(sp))
    p["m.E"] = E
    Cw = torch.zeros(maps.mod.C.weight.shape, dtype=torch.float64)
    Cw[:, :8] = C_t / sc
    p["C.weight"] = Cw
    return flat(p, maps.tkeys), f"z* = {zstar} (têtes à porte), r_max = {float(r.max()):.4f}"


# ------------------------------------------------------------------------------------------
def job(kind):
    torch.set_num_threads(1)
    lr = KINDS[kind]
    torch.manual_seed(SEED)
    A, B, C = osc.teacher(SEED)
    u_tr, u_te = osc.inputs(256, 100, 10 + SEED), osc.inputs(128, 300, 20 + SEED)
    y_tr, y_te = osc.sim(A, B, C, u_tr), osc.sim(A, B, C, u_te)
    sc = y_tr.std()
    y_tr, y_te = y_tr / sc, y_te / sc
    model = osc.make(kind)
    maps = RenMaps(model) if kind == "ren" else BlockMaps(model)
    ud, yd = u_tr.double(), y_tr.double()

    # contrôle 1 : la perte en W reproduit la perte du module
    with torch.no_grad():
        l_mod = float(((model(u_tr) - y_tr) ** 2).mean())
        l_W = float(maps.loss_W(maps.phi(maps.theta().double()), ud, yd))
    checks = dict(loss_module=l_mod, loss_W=l_W, fd_rel_err=fd_check(maps, maps.theta()))

    # entraînement identique à oscillator.run (Adam, clip 10), instantanés de θ
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    snaps, t0 = [(0, maps.theta().clone())], time.time()
    for it in range(1, ITERS + 1):
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if it % EVERY == 0:
            snaps.append((it, maps.theta().clone()))
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean())
    t_train = time.time() - t0

    traj = []
    for it, th in snaps:
        d = diagnose(maps, th, ud, yd)
        d["iter"] = it
        traj.append(d)

    star = None
    if kind != "ren":
        th_s, note = teacher_theta(maps, kind, float(sc))
        star = dict(note=note)
        if th_s is not None:
            # point générique voisin : la norme spectrale n'est pas dérivable en un bloc r·R(φ)
            # (σ₁ = σ₂) ni en un bloc nul ; on mesure J à 1e-6 près du maître.
            g = torch.Generator().manual_seed(1)
            th_s = th_s + 1e-6 * torch.randn(th_s.shape, generator=g, dtype=th_s.dtype)
            star.update(diagnose(maps, th_s, ud, yd))
            star["fd_rel_err"] = fd_check(maps, th_s)
    res = dict(kind=kind, lr=lr, seed=SEED, test=test, t_train=t_train, checks=checks, traj=traj, star=star)
    json.dump(res, open(os.path.join(OUT, f"m1_{kind}.json"), "w"), indent=1)
    return f"{kind}: test={test:.4f} train {t_train:.0f}s, fd={checks['fd_rel_err']:.1e}"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    kinds = os.environ.get("KINDS", ",".join(KINDS)).split(",")
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in pool.imap_unordered(job, kinds):
            print(line, flush=True)
