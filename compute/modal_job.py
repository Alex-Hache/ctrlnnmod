"""Lance une commande du dépôt sur un GPU Modal et rapatrie $OUT_DIR.

Usage (dans un job GitHub Actions, secrets MODAL_TOKEN_ID / MODAL_TOKEN_SECRET) :
    MODAL_GPU=L4 modal run compute/modal_job.py --cmd "python experiments/..."
"""
import io
import os
import subprocess
import tarfile

import modal

GPU = os.environ.get("MODAL_GPU", "L4")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("torch", "numpy", "scipy", "matplotlib", "mujoco", "cvxpy", "typeguard",
                      "lightning", "tensorboard")
         .add_local_dir(".", "/root/repo", ignore=[".git", "out", "**/__pycache__", "*.pt"]))
app = modal.App("ctrlnnmod-compute")


@app.function(image=image, gpu=GPU, timeout=6 * 3600)
def run(cmd: str) -> bytes:
    os.chdir("/root/repo")
    out = "/root/repo/out_modal"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OUT_DIR=out, NTHREADS="8",
               PYTHONPATH=".:experiments/closed_loop:experiments/world_model")
    p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    with open(f"{out}/log_modal.txt", "w") as f:
        f.write(f"$ {cmd}\nexit={p.returncode}\n{p.stdout}\n{p.stderr}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main(cmd: str):
    data = run.remote(cmd)
    dest = os.environ.get("OUT_DIR", "out/modal")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)
    print(open(f"{dest}/log_modal.txt").read()[-3000:])
