r"""
Noyau Triton du REN sélectif : boucle temporelle persistante, avant et arrière.

Récurrence (par tête h, vecteurs ligne comme dans le module PyTorch) :

    pre_t = x_t C1ᵀ + P_t            v_t = pre_t / λ        w_t = tanh(v_t)
    f_t   = x_t Aᵀ + w_t Bwᵀ + Q_t   x_{t+1} = (1 − g_t) x_t + g_t f_t

P_t = D12 u_t + b_v, Q_t = B_u u_t + b_x et g_t ne dépendent pas de l'état : ils sont calculés
pour toute la séquence en dehors du noyau (grands produits matriciels). Le noyau ne fait que
la partie séquentielle. Un programme = (bloc de BB trajectoires, une tête) ; l'état reste sur
la puce pendant les T pas.

Arrière : un second noyau parcourt le temps à l'envers et calcule l'adjoint
a_t = ∂L/∂x_t ; il écrit dQ_t = g a, dP_t = ∂L/∂pre_t, dg_t. Les gradients des poids sont
ensuite des sommes sur (b, t) : un seul einsum chacun, en PyTorch.

Contraintes Triton : BB, NX, NQ puissances de 2 et ≥ 16 (tl.dot).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(v):
    return 2.0 * tl.sigmoid(2.0 * v) - 1.0


@triton.jit
def _sren_fwd(x0, P, Q, G, AT, C1T, BwT, LAM, X, V,
              B, T, H,
              BB: tl.constexpr, NX: tl.constexpr, NQ: tl.constexpr, PREC: tl.constexpr):
    pb = tl.program_id(0)
    h = tl.program_id(1)
    rb = pb * BB + tl.arange(0, BB)
    mb = rb < B
    ix = tl.arange(0, NX)
    iq = tl.arange(0, NQ)
    # poids de la tête h (constants sur la séquence) : chargés une fois
    at = tl.load(AT + h * NX * NX + ix[:, None] * NX + ix[None, :])
    c1t = tl.load(C1T + h * NX * NQ + ix[:, None] * NQ + iq[None, :])
    bwt = tl.load(BwT + h * NQ * NX + iq[:, None] * NX + ix[None, :])
    lam = tl.load(LAM + h * NQ + iq)
    x = tl.load(x0 + (rb[:, None] * H + h) * NX + ix[None, :], mask=mb[:, None], other=0.0)
    tl.store(X + ((rb[:, None] * (T + 1)) * H + h) * NX + ix[None, :], x, mask=mb[:, None])
    for t in range(T):
        row = (rb * T + t) * H + h
        p = tl.load(P + row[:, None] * NQ + iq[None, :], mask=mb[:, None], other=0.0)
        q = tl.load(Q + row[:, None] * NX + ix[None, :], mask=mb[:, None], other=0.0)
        g = tl.load(G + row, mask=mb, other=0.0)
        v = (tl.dot(x, c1t, input_precision=PREC) + p) / lam[None, :]
        w = _tanh(v)
        f = tl.dot(x, at, input_precision=PREC) + tl.dot(w, bwt, input_precision=PREC) + q
        x = (1.0 - g)[:, None] * x + g[:, None] * f
        tl.store(X + ((rb[:, None] * (T + 1) + t + 1) * H + h) * NX + ix[None, :], x, mask=mb[:, None])
        tl.store(V + row[:, None] * NQ + iq[None, :], v, mask=mb[:, None])


@triton.jit
def _sren_bwd(X, V, Q, G, dX, AT, C1T, BwT, LAM, dP, dQ, dG, dx0,
              B, T, H,
              BB: tl.constexpr, NX: tl.constexpr, NQ: tl.constexpr, PREC: tl.constexpr):
    pb = tl.program_id(0)
    h = tl.program_id(1)
    rb = pb * BB + tl.arange(0, BB)
    mb = rb < B
    ix = tl.arange(0, NX)
    iq = tl.arange(0, NQ)
    at = tl.load(AT + h * NX * NX + ix[:, None] * NX + ix[None, :])          # Aᵀ  [NX, NX]
    c1t = tl.load(C1T + h * NX * NQ + ix[:, None] * NQ + iq[None, :])        # C1ᵀ [NX, NQ]
    bwt = tl.load(BwT + h * NQ * NX + iq[:, None] * NX + ix[None, :])        # Bwᵀ [NQ, NX]
    a_ = tl.load(AT + h * NX * NX + ix[None, :] * NX + ix[:, None])          # A   [NX, NX]
    c1 = tl.load(C1T + h * NX * NQ + ix[None, :] * NQ + iq[:, None])         # C1  [NQ, NX]
    bw = tl.load(BwT + h * NQ * NX + iq[None, :] * NX + ix[:, None])         # Bw  [NX, NQ]
    lam = tl.load(LAM + h * NQ + iq)
    a = tl.zeros((BB, NX), dtype=tl.float32)
    for k in range(T):
        t = T - 1 - k
        row = (rb * T + t) * H + h
        a += tl.load(dX + row[:, None] * NX + ix[None, :], mask=mb[:, None], other=0.0)
        x = tl.load(X + ((rb[:, None] * (T + 1) + t) * H + h) * NX + ix[None, :], mask=mb[:, None], other=0.0)
        v = tl.load(V + row[:, None] * NQ + iq[None, :], mask=mb[:, None], other=0.0)
        q = tl.load(Q + row[:, None] * NX + ix[None, :], mask=mb[:, None], other=0.0)
        g = tl.load(G + row, mask=mb, other=0.0)
        w = _tanh(v)
        f = tl.dot(x, at, input_precision=PREC) + tl.dot(w, bwt, input_precision=PREC) + q
        dg = tl.sum(a * (f - x), axis=1)
        df = g[:, None] * a
        dw = tl.dot(df, bw, input_precision=PREC)
        dpre = dw * (1.0 - w * w) / lam[None, :]
        a = (1.0 - g)[:, None] * a + tl.dot(df, a_, input_precision=PREC) + tl.dot(dpre, c1, input_precision=PREC)
        tl.store(dQ + row[:, None] * NX + ix[None, :], df, mask=mb[:, None])
        tl.store(dP + row[:, None] * NQ + iq[None, :], dpre, mask=mb[:, None])
        tl.store(dG + row, dg, mask=mb)
    tl.store(dx0 + (rb[:, None] * H + h) * NX + ix[None, :], a, mask=mb[:, None])


def _block(B):
    return 16 if B <= 16 else 32


class SRENScan(torch.autograd.Function):
    """x0 [B,H,NX], P [B,T,H,NQ], Q [B,T,H,NX], G [B,T,H], AT [H,NX,NX], C1T [H,NX,NQ],
    BwT [H,NQ,NX], lam [H,NQ] → X [B,T+1,H,NX] (états, x0 compris)."""

    @staticmethod
    def forward(ctx, x0, P, Q, G, AT, C1T, BwT, lam, prec="ieee"):
        B, T, H, NQ = P.shape
        NX = Q.shape[-1]
        x0, P, Q, G = (t.contiguous().float() for t in (x0, P, Q, G))
        AT, C1T, BwT, lam = (t.contiguous().float() for t in (AT, C1T, BwT, lam))
        X = torch.empty(B, T + 1, H, NX, device=P.device)
        V = torch.empty(B, T, H, NQ, device=P.device)
        BB = _block(B)
        _sren_fwd[(triton.cdiv(B, BB), H)](x0, P, Q, G, AT, C1T, BwT, lam, X, V, B, T, H,
                                           BB=BB, NX=NX, NQ=NQ, PREC=prec)
        ctx.save_for_backward(X, V, Q, G, AT, C1T, BwT, lam)
        ctx.prec = prec
        return X

    @staticmethod
    def backward(ctx, dXall):
        X, V, Q, G, AT, C1T, BwT, lam = ctx.saved_tensors
        B, T, H, NQ = V.shape
        NX = X.shape[-1]
        dXall = dXall.contiguous().float()
        dX = dXall[:, 1:].contiguous()
        dP = torch.empty_like(V)
        dQ = torch.empty_like(Q)
        dG = torch.empty_like(G)
        dx0 = torch.empty(B, H, NX, device=V.device)
        BB = _block(B)
        _sren_bwd[(triton.cdiv(B, BB), H)](X, V, Q, G, dX, AT, C1T, BwT, lam, dP, dQ, dG, dx0, B, T, H,
                                           BB=BB, NX=NX, NQ=NQ, PREC=ctx.prec)
        dx0 = dx0 + dXall[:, 0]
        Xp = X[:, :-1]
        W = torch.tanh(V)
        dAT = torch.einsum("bthi,bthj->hij", Xp, dQ)
        dBwT = torch.einsum("bthk,bthj->hkj", W, dQ)
        dC1T = torch.einsum("bthi,bthk->hik", Xp, dP)
        dlam = -torch.einsum("bthk,bthk->hk", dP, V)
        return dx0, dP, dQ, dG, dAT, dC1T, dBwT, dlam, None


def sren_scan(x0, P, Q, G, AT, C1T, BwT, lam, prec="ieee"):
    # Les poids d'une tête restent sur la puce pendant tout le balayage : A, C1, Bw en fp32.
    # Mémoire partagée ≈ 100 Ko par SM (L4, A10) : on garde nx·nx + 2·nx·nq ≤ 8192 flottants
    # (nx = 32, nq = 64 : 5120). Plus large : plus de têtes, pas des têtes plus grandes.
    NX, NQ = AT.shape[-1], C1T.shape[-1]
    if NX * NX + 2 * NX * NQ > 8192:
        raise ValueError(f"tête trop large pour le noyau (nx={NX}, nq={NQ}) : "
                         "garder nx ≤ 32 et nq ≤ 64, et augmenter n_heads")
    return SRENScan.apply(x0, P, Q, G, AT, C1T, BwT, lam, prec)
