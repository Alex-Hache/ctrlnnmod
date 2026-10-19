"""Lance une commande du dépôt sur Modal (GPU ou CPU) et rapatrie $OUT_DIR.

    MODAL_GPU=L4 modal run compute/modal_job.py --cmd "python experiments/..."
    MODAL_GPU=none modal run compute/modal_job.py --cmd "..."      # CPU seulement (8 cœurs)

Le volume Modal `ctrlnnmod-data` est monté sur /data (données partagées entre jobs).
Rendu MuJoCo hors écran : OSMesa (MUJOCO_GL=osmesa).
"""
import io
import os
import subprocess
import tarfile

import modal

GPU = os.environ.get("MODAL_GPU", "L4")
MENAGERIE = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda")

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "libosmesa6", "libgl1", "libglib2.0-0")
         .pip_install("torch", "numpy", "scipy", "matplotlib", "mujoco", "cvxpy", "typeguard",
                      "lightning", "tensorboard", "PyOpenGL")
         .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
               "MENAGERIE": "/root/mujoco_menagerie"}))
if os.path.isdir(MENAGERIE):
    image = image.add_local_dir(MENAGERIE, "/root/mujoco_menagerie/franka_emika_panda")
image = image.add_local_dir(".", "/root/repo", ignore=[".git", "out", "**/__pycache__", "*.pt",
                                                       "**/results_ci/*.pt"])
app = modal.App("ctrlnnmod-compute")
vol = modal.Volume.from_name("ctrlnnmod-data", create_if_missing=True)


def _run(cmd: str) -> bytes:
    os.chdir("/root/repo")
    out = "/root/repo/out_modal"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OUT_DIR=out, NTHREADS="8",
               PYTHONPATH=".:experiments/closed_loop:experiments/world_model:experiments/pixel_wm")
    p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    with open(f"{out}/log_modal.txt", "w") as f:
        f.write(f"$ {cmd}\nexit={p.returncode}\n{p.stdout}\n{p.stderr}")
    vol.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    return buf.getvalue()


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=8.0, memory=32768,
              timeout=8 * 3600, volumes={"/data": vol})
def run(cmd: str) -> bytes:
    return _run(cmd)


@app.local_entrypoint()
def main(cmd: str):
    data = run.remote(cmd)
    dest = os.environ.get("OUT_DIR", "out/modal")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")
    print(open(f"{dest}/log_modal.txt").read()[-4000:])
