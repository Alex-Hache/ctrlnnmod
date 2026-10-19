"""
M3a — modèles de référence de l'état de l'art, sur la même tâche que pwm.py.

Chaque modèle reçoit deux images 64×64 (t, t+1) et les actions, et prédit l'état articulaire
futur (18 grandeurs normalisées). Interface commune :

    init_state(img) → état            advance(état, a_n) → (état, ô_n)
    forward(img, a) → ô (B, T, 18)    train_loss(data, B, H, gen) → perte

ô_n a la forme (B, 18), ou (K, B, 18) pour un ensemble (la politique voit la moyenne, la
perte d'imagination porte sur chaque membre).

Modèles (noms courts, sans « _ ») :
- rssm    : RSSM type DreamerV3 (GRU + latents catégoriels 16×16, KL équilibré, free bits 1 nat,
            unimix 1 %, gradients straight-through). Entraînement avec postérieur sur o_t.
- ens     : ensemble de 5 GRU (PETS / MBPO). Lots indépendants par membre.
- simnorm : dynamique latente type TD-MPC2 (MLP + LayerNorm + Mish, latent SimNorm 512).
- tfm     : transformer causal type STORM / IRIS (RoPE, fenêtre 50, apprentissage forcé par
            l'enseignant, déroulés autorégressifs avec cache clé-valeur).
- lru     : Linear Recurrent Unit (Orvieto et al., 2023), stable par construction, 2 couches.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from pwm import NA, NO, Encoder, dev, mlp


class Base(nn.Module):
    """forward et perte par défaut : déroulé en boucle ouverte, erreur quadratique."""

    def forward(self, img, a):
        st = self.init_state(img)
        outs = []
        for t in range(a.shape[1]):
            st, o = self.advance(st, a[:, t])
            outs.append(o.mean(0) if o.dim() == 3 else o)
        return torch.stack(outs, 1)

    def train_loss(self, data, B, H, gen):
        img, a, o = data.batch(B, H, gen)
        return ((self(img, a) - o) ** 2).mean()


# --------------------------------------------------------------------------- RSSM (DreamerV3)
class RSSM(Base):
    def __init__(self, n=256, groups=16, classes=16, free=1.0):
        super().__init__()
        self.G, self.C, self.free = groups, classes, free
        S = groups * classes
        self.enc = Encoder(n)
        self.img_in = nn.Sequential(nn.Linear(S + NA, n), nn.LayerNorm(n), nn.SiLU())
        self.cell = nn.GRUCell(n, n)
        self.prior = nn.Sequential(nn.Linear(n, n), nn.LayerNorm(n), nn.SiLU(), nn.Linear(n, S))
        self.post = nn.Sequential(nn.Linear(n + NO, n), nn.LayerNorm(n), nn.SiLU(), nn.Linear(n, S))
        self.dec = mlp(n + S, 256, NO)

    def _probs(self, logits):
        p = logits.view(*logits.shape[:-1], self.G, self.C).softmax(-1)
        return 0.99 * p + 0.01 / self.C                                     # unimix 1 %

    @staticmethod
    def _sample(p):
        idx = torch.multinomial(p.reshape(-1, p.shape[-1]), 1).view(p.shape[:-1])
        oh = F.one_hot(idx, p.shape[-1]).to(p.dtype)
        return (oh + p - p.detach()).flatten(-2)                            # straight-through

    @staticmethod
    def _kl(p, q):
        return (p * (p.log() - q.log())).sum((-1, -2))

    def init_state(self, img):
        h = self.enc(img)
        return h, self._sample(self._probs(self.prior(h)))

    def advance(self, st, an):
        h, z = st
        h = self.cell(self.img_in(torch.cat([z, an], -1)), h)
        z = self._sample(self._probs(self.prior(h)))
        return (h, z), self.dec(torch.cat([h, z], -1))

    def train_loss(self, data, B, H, gen):
        img, a, o = data.batch(B, H, gen)
        h, z = self.init_state(img)
        rec = dyn = rep = 0.0
        for t in range(H):
            h = self.cell(self.img_in(torch.cat([z, a[:, t]], -1)), h)
            pq = self._probs(self.prior(h))
            pp = self._probs(self.post(torch.cat([h, o[:, t]], -1)))
            z = self._sample(pp)
            rec = rec + ((self.dec(torch.cat([h, z], -1)) - o[:, t]) ** 2).sum(-1).mean()
            dyn = dyn + self._kl(pp.detach(), pq).clamp(min=self.free).mean()
            rep = rep + self._kl(pp, pq.detach()).clamp(min=self.free).mean()
        return (rec + 0.5 * dyn + 0.1 * rep) / H


# --------------------------------------------------------------------------- ensemble (PETS)
class GRUMember(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc, self.cell, self.dec = Encoder(n), nn.GRU(NA, n, batch_first=True), mlp(n, 256, NO)

    def forward(self, img, a):
        xs, _ = self.cell(a, self.enc(img)[None].contiguous())
        return self.dec(xs)


class Ensemble(Base):
    def __init__(self, n=256, K=5):
        super().__init__()
        self.members = nn.ModuleList([GRUMember(n) for _ in range(K)])

    def forward(self, img, a):
        return torch.stack([m(img, a) for m in self.members]).mean(0)

    def train_loss(self, data, B, H, gen):
        loss = 0.0
        for m in self.members:                                              # un lot par membre
            img, a, o = data.batch(B, H, gen)
            loss = loss + ((m(img, a) - o) ** 2).mean()
        return loss / len(self.members)

    def init_state(self, img):
        return [m.enc(img)[None].contiguous() for m in self.members]

    def advance(self, st, an):
        new, outs = [], []
        for m, h in zip(self.members, st):
            y, h = m.cell(an[:, None], h)
            new.append(h)
            outs.append(m.dec(y[:, 0]))
        return new, torch.stack(outs)                                       # (K, B, 18)


# --------------------------------------------------------------------------- TD-MPC2 (SimNorm)
def simnorm(x, g=8):
    return x.view(*x.shape[:-1], -1, g).softmax(-1).flatten(-2)


class SimNormWM(Base):
    def __init__(self, n=256, latent=512):
        super().__init__()
        self.enc = Encoder(latent)
        self.dyn = nn.Sequential(nn.Linear(latent + NA, 512), nn.LayerNorm(512), nn.Mish(),
                                 nn.Linear(512, 512), nn.LayerNorm(512), nn.Mish(), nn.Linear(512, latent))
        self.dec = mlp(latent, 256, NO)

    def init_state(self, img):
        return simnorm(self.enc(img))

    def advance(self, z, an):
        z = simnorm(self.dyn(torch.cat([z, an], -1)))
        return z, self.dec(z)


# --------------------------------------------------------------------------- transformer (STORM)
def rope(x, pos, base=10000.0):
    """x : (B, h, L, dh), pos : (L,) entiers. Rotation par paires (RoPE)."""
    dh = x.shape[-1]
    inv = base ** (-torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh)
    ang = pos.to(x.dtype)[:, None] * inv[None]                              # (L, dh/2)
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class TBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _qkv(self, x, pos):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        return rope(q, pos), rope(k, pos), v

    def forward(self, x, pos):                                              # séquence entière
        q, k, v = self._qkv(x, pos)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).flatten(2))
        return x + self.ff(self.ln2(x))

    def step(self, x, pos, cache, W):                                       # un jeton, cache (K, V)
        q, k, v = self._qkv(x, pos)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2)[:, :, -W:], torch.cat([cache[1], v], 2)[:, :, -W:]
        y = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(y.transpose(1, 2).flatten(2))
        return x + self.ff(self.ln2(x)), (k, v)


class TransformerWM(Base):
    def __init__(self, n=256, layers=3, heads=4, window=50):
        super().__init__()
        self.W = window
        self.enc, self.emb_o, self.emb_a = Encoder(n), nn.Linear(NO, n), nn.Linear(NA, n)
        self.blocks = nn.ModuleList([TBlock(n, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(n)
        self.head = mlp(n, 256, NO)

    def _run(self, tok):
        pos = torch.arange(tok.shape[1], device=tok.device)
        for b in self.blocks:
            tok = b(tok, pos)
        return self.head(self.ln(tok))

    def train_loss(self, data, B, H, gen):
        # moitié des fenêtres commence aux images, moitié en milieu de trajectoire (jeton d'état) :
        # le modèle apprend à continuer quand l'image sort de la fenêtre glissante
        img, a, o = data.batch(B // 2, H, gen)
        t_img = torch.cat([self.enc(img)[:, None], self.emb_o(o[:, :-1])], 1) + self.emb_a(a)
        O, A, _ = data.tr
        i = torch.randint(0, O.shape[0], (B - B // 2,), generator=gen)
        t0 = torch.randint(0, A.shape[1] - H + 1, (B - B // 2,), generator=gen)
        idx = t0[:, None] + torch.arange(H + 1)
        Ow, Aw = O[i[:, None], idx].to(dev), A[i[:, None], idx[:, :-1]].to(dev)
        t_obs = self.emb_o(Ow[:, :-1]) + self.emb_a(Aw)
        pred = self._run(torch.cat([t_img, t_obs]))
        return ((pred - torch.cat([o, Ow[:, 1:]])) ** 2).mean()

    def init_state(self, img):
        return dict(first=self.enc(img), prev=None, pos=0, cache=[None] * len(self.blocks))

    def advance(self, st, an):
        tok = (st["first"] if st["prev"] is None else self.emb_o(st["prev"])) + self.emb_a(an)
        x, pos, cache = tok[:, None], torch.tensor([st["pos"]], device=an.device), []
        for b, c in zip(self.blocks, st["cache"]):
            x, c = b.step(x, pos, c, self.W)
            cache.append(c)
        o = self.head(self.ln(x[:, 0]))
        return dict(first=st["first"], prev=o, pos=st["pos"] + 1, cache=cache), o


# --------------------------------------------------------------------------- LRU
class LRULayer(nn.Module):
    """x_t = λ ⊙ x_{t−1} + γ ⊙ B u_t (complexe, diagonal), y = Re(C x) + D u, |λ| < 1 par construction."""

    def __init__(self, d, N, r_min=0.5, r_max=0.99, max_phase=math.pi / 10):
        super().__init__()
        u1, u2 = torch.rand(N), torch.rand(N)
        nu = -0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)
        self.nu_log, self.theta_log = nn.Parameter(nu.log()), nn.Parameter((max_phase * u2).log())
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - torch.exp(-2 * nu))))
        self.B_re, self.B_im = (nn.Parameter(torch.randn(N, d) / math.sqrt(2 * d)) for _ in range(2))
        self.C_re, self.C_im = (nn.Parameter(torch.randn(d, N) / math.sqrt(N)) for _ in range(2))
        self.D = nn.Parameter(torch.randn(d) * 0.1)
        self.ln = nn.LayerNorm(d)
        self.glu = nn.Linear(d, 2 * d)

    def step(self, st, u):
        xr, xi = st
        r, th, g = torch.exp(-self.nu_log.exp()), self.theta_log.exp(), self.gamma_log.exp()
        lr, li = r * th.cos(), r * th.sin()
        v = self.ln(u)
        br, bi = v @ self.B_re.T, v @ self.B_im.T
        xr, xi = lr * xr - li * xi + g * br, lr * xi + li * xr + g * bi
        y = xr @ self.C_re.T - xi @ self.C_im.T + self.D * v
        return (xr, xi), u + F.glu(self.glu(F.gelu(y)), -1)


class LRUWM(Base):
    def __init__(self, n=256, N=256, layers=2):
        super().__init__()
        self.N = N
        self.enc, self.act_emb = Encoder(n), nn.Linear(NA, n)
        self.layers = nn.ModuleList([LRULayer(n, N) for _ in range(layers)])
        self.dec = mlp(n, 256, NO)

    def _run(self, sts, tok):
        new = []
        for lay, st in zip(self.layers, sts):
            st, tok = lay.step(st, tok)
            new.append(st)
        return new, tok

    def init_state(self, img):
        B = img.shape[0]
        z = torch.zeros(B, self.N, device=img.device)
        sts, _ = self._run([(z, z) for _ in self.layers], self.enc(img))
        return sts

    def advance(self, sts, an):
        sts, tok = self._run(sts, self.act_emb(an))
        return sts, self.dec(tok)


BASELINES = {"rssm": RSSM, "ens": Ensemble, "simnorm": SimNormWM, "tfm": TransformerWM, "lru": LRUWM}
