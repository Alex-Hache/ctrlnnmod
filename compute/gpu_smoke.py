"""Test GPU : périphérique, puis temps d'une itération (batch 64, H = 100) à grande largeur."""
import json
import os
import subprocess
import time

import torch

from ctrlnmod.models.discrete.explicit_ren import ExplicitREN
from ctrlnmod.models.discrete.selective_ren import SelectiveREN

out = os.environ.get("OUT_DIR", ".")
dev = "cuda" if torch.cuda.is_available() else "cpu"
res = {"device": dev, "name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
       "nvidia_smi": subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True,
                                    capture_output=True, text=True).stdout}
print(res)
B, H, NA = 64, 100, 8
for n in (256, 1024, 2048):
    for name, mk in (("gru", lambda: torch.nn.GRUCell(NA, n)),
                     ("ren_explicit", lambda: ExplicitREN(NA, 1, n, n)),
                     ("sren", lambda: SelectiveREN(NA, n_heads=8, nx=n // 8, nq=2 * (n // 8)))):
        m = mk().to(dev)
        a = torch.randn(B, H, NA, device=dev)
        def it():
            s = 0
            if name == "gru":
                x = torch.zeros(B, n, device=dev)
                for t in range(H):
                    x = m(a[:, t], x); s = s + x.square().mean()
            elif name == "ren_explicit":
                fr = m.frame(); x = torch.zeros(B, n, device=dev)
                for t in range(H):
                    x, _ = m.step(fr, x, a[:, t]); s = s + x.square().mean()
            else:
                fr = m.frame(); x = torch.zeros(B, m.n_state, device=dev)
                for t in range(H):
                    x = m.step(fr, x, a[:, t]); s = s + x.square().mean()
            s.backward()
        it()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            it()
        if dev == "cuda":
            torch.cuda.synchronize()
        res[f"{name}/{n}"] = (time.perf_counter() - t0) / 3
        print(name, n, f"{res[f'{name}/{n}']:.3f} s/it", flush=True)
json.dump(res, open(f"{out}/gpu_smoke.json", "w"), indent=1)
