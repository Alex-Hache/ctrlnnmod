"""Banc GPU : temps d'une itération d'entraînement (avant + arrière), batch 64, T = 100.

Variantes du REN sélectif : pas du module en eager, boucle restructurée en eager, torch.compile
(défaut), torch.compile(mode="reduce-overhead") = CUDA graphs, noyau Triton (fp32 IEEE et TF32).
Références : nn.GRU (cuDNN, séquence entière en un noyau) et GRUCell en boucle.
"""
import json
import os
import statistics
import time
import traceback

import torch
from tqdm import tqdm

from ctrlnmod.models.discrete.selective_ren import SelectiveREN, rollout

out = os.environ.get("OUT_DIR", ".")
dev = "cuda"
B, T, NU = 64, 100, 8
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def timeit(fn, reps=5):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = time.perf_counter() - t0          # inclut la compilation / la capture
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), first


for n in (128, 512, 1024, 2048):
    nx = n // 8
    torch.manual_seed(0)
    m = SelectiveREN(NU, n_heads=8, nx=nx, nq=2 * nx).to(dev)
    u = torch.randn(B, T, NU, device=dev)
    Wt = torch.randn(B, T, n, device=dev)

    def ref_states():
        with torch.no_grad():
            return rollout(m, u, None, "torch")
    ref = ref_states()

    def make(kind):
        if kind == "step_eager":
            def f():
                fr = m.frame(); x = torch.zeros(B, m.n_state, device=dev); s = 0
                for t in range(T):
                    x = m.step(fr, x, u[:, t]); s = s + (x * Wt[:, t]).sum()
                s.backward()
            return f, None
        if kind in ("rollout_eager", "compile_default", "compile_cudagraphs"):
            def loss():
                return (rollout(m, u, None, "torch") * Wt).sum()
            if kind == "compile_default":
                loss = torch.compile(loss)
            if kind == "compile_cudagraphs":
                loss = torch.compile(loss, mode="reduce-overhead")
            def f():
                l = loss(); l.backward()
            check = lambda: rollout(m, u, None, "torch")  # noqa: E731
            return f, None
        if kind.startswith("triton"):
            prec = "tf32" if kind.endswith("tf32") else "ieee"
            def f():
                l = (rollout(m, u, None, "triton", prec) * Wt).sum(); l.backward()
            def check():
                with torch.no_grad():
                    return rollout(m, u, None, "triton", prec)
            return f, check
        if kind == "gru_cudnn":
            g = torch.nn.GRU(NU, n, batch_first=True).to(dev)
            def f():
                y, _ = g(u); (y * Wt).sum().backward()
            return f, None
        if kind == "gru_cell_loop":
            c = torch.nn.GRUCell(NU, n).to(dev)
            def f():
                x = torch.zeros(B, n, device=dev); s = 0
                for t in range(T):
                    x = c(u[:, t], x); s = s + (x * Wt[:, t]).sum()
                s.backward()
            return f, None

    for kind in tqdm(("gru_cudnn", "gru_cell_loop", "step_eager", "rollout_eager", "compile_default",
                      "compile_cudagraphs", "triton_ieee", "triton_tf32"), desc=f"n={n}", ncols=90):
        key = f"{kind}/{n}"
        try:
            torch._dynamo.reset()
            f, check = make(kind)
            t, first = timeit(f)
            res[key] = {"s_per_iter": t, "first_call_s": first}
            if check is not None:
                res[key]["max_err_vs_eager"] = (check() - ref).abs().max().item()
            print(key, res[key], flush=True)
        except Exception as e:  # noqa: BLE001
            res[key] = {"error": repr(e)[:300]}
            print(key, "ERREUR", repr(e)[:300], flush=True)
            traceback.print_exc()
        json.dump(res, open(f"{out}/bench_kernels.json", "w"), indent=1)
