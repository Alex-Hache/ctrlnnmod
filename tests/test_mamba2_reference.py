"""Le bloc Mamba-2 de experiments/world_model/wm.py est identique au bloc de référence de
Hugging Face transformers (Mamba2Block, chemin PyTorch), à la précision machine.

HF calcule certaines étapes en float32 même en float64 (`.float()`, `.to(torch.float32)`) ;
le test retire ces conversions pour comparer en float64 de bout en bout.
"""
import inspect
import re
import sys

import pytest
import torch

mm = pytest.importorskip("transformers.models.mamba2.modeling_mamba2")
sys.path.insert(0, "experiments/world_model")


def _strip(src):
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("@"))
    return re.sub(r"\.float\(\)|\.to\(torch\.float32\)", "", src)


def _float64_hf():
    ns = {}
    exec(_strip(inspect.getsource(mm.mamba2_chunk_scan)), mm.__dict__, ns)
    mm.mamba2_chunk_scan = ns["mamba2_chunk_scan"]
    for cls in (mm.MambaRMSNormGated, mm.Mamba2RMSNorm, mm.Mamba2Mixer):
        src = "\n".join(l[4:] for l in _strip(inspect.getsource(cls.forward)).splitlines())
        ns = {}
        exec(src, mm.__dict__, ns)
        cls.forward = ns["forward"]


def test_mamba2_block_matches_hf():
    torch.set_default_dtype(torch.float64)
    try:
        _float64_hf()
        from transformers import Mamba2Config
        from wm import Mamba2Block
        torch.manual_seed(0)
        d = 96
        mine = Mamba2Block(d)
        with torch.no_grad():
            for p in mine.parameters():
                p.add_(0.3 * torch.randn_like(p))
        cfg = Mamba2Config(hidden_size=d, expand=2, num_heads=8, head_dim=24, state_size=16, n_groups=1,
                           conv_kernel=4, use_conv_bias=True, use_bias=True, layer_norm_epsilon=1e-6,
                           chunk_size=64, num_hidden_layers=1, time_step_limit=(0.0, float("inf")),
                           residual_in_fp32=False)
        hf = mm.Mamba2Block(cfg, 0).double().eval()
        with torch.no_grad():
            m = hf.mixer
            m.in_proj.weight.copy_(mine.in_proj.weight)
            m.in_proj.bias.copy_(mine.in_proj.bias)
            m.conv1d.weight.copy_(mine.conv_w[:, None, :])
            m.conv1d.bias.copy_(mine.conv_b)
            m.dt_bias.copy_(mine.dt_bias)
            m.A_log.copy_(mine.A_log)
            m.D.copy_(mine.Dskip)
            m.norm.weight.copy_(mine.norm_w)
            m.out_proj.weight.copy_(mine.out_proj.weight)
            m.out_proj.bias.copy_(mine.out_proj.bias)
            hf.norm.weight.copy_(mine.pre_norm)
            x = torch.randn(4, 200, d)
            y_hf = hf(x)
            st, ys = mine.init_state(4), []
            for t in range(200):
                st, y = mine.step(st, x[:, t])
                ys.append(y)
            assert (y_hf - torch.stack(ys, 1)).abs().max() < 1e-10
    finally:
        torch.set_default_dtype(torch.float32)
