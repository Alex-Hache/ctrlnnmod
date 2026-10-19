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
MENAGERIE = next((p for p in (os.environ.get("MENAGERIE_LOCAL", ""), "/home/claude/mujoco_menagerie/franka_emika_panda",
                   os.path.expanduser("~/mujoco_menagerie/franka_emika_panda")) if p and os.path.isdir(p)), "")

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "libosmesa6", "libgl1", "libglib2.0-0")
         .pip_install("torch", "numpy", "scipy", "matplotlib", "mujoco", "cvxpy", "typeguard",
                      "lightning", "tensorboard", "PyOpenGL", "tqdm")
         .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
               "MENAGERIE": "/root/mujoco_menagerie"}))
if MENAGERIE:
    image = image.add_local_dir(MENAGERIE, "/root/mujoco_menagerie/franka_emika_panda")
image = image.add_local_dir(".", "/root/repo", ignore=[".git", "out", "**/__pycache__", "*.pt",
                                                       "**/results_ci/*.pt"])
RUN_NAME = os.environ.get("RUN_NAME", "compute")
app = modal.App(f"ctrlnnmod-{RUN_NAME}")          # un nom par run, visible dans le tableau de bord
vol = modal.Volume.from_name("ctrlnnmod-data", create_if_missing=True)


def _run(cmd: str, label: str = "") -> bytes:
    """Exécute cmd ; chaque ligne de sortie part tout de suite dans les logs Modal (et dans un fichier)."""
    import sys
    import time
    os.chdir("/root/repo")
    out = "/root/repo/out_modal"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OUT_DIR=out, NTHREADS="8", NPROC="8", PYTHONUNBUFFERED="1",
               PYTHONPATH=".:experiments/closed_loop:experiments/world_model:experiments/pixel_wm")
    t0 = time.time()
    print(f"[{label}] $ {cmd}", flush=True)
    with open(f"{out}/log_modal.txt", "wb") as f:
        f.write(f"$ {cmd}\n".encode())
        # sortie brute : les \r des barres de progression (tqdm) passent tels quels
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            chunk = p.stdout.read1(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            f.write(chunk)
        p.wait()
        f.write(f"exit={p.returncode}\n".encode())
    print(f"[{label}] fini, exit={p.returncode}, {time.time() - t0:.0f}s", flush=True)
    # copie complète sur le volume ; retour au client limité aux petits fichiers (< 1 Mo) :
    # les gros retours passent par un stockage objet que le proxy du bac à sable refuse
    import shutil
    dst = f"/data/runs/{label.split(' ')[0] or 'run'}"
    shutil.copytree(out, dst, dirs_exist_ok=True)
    vol.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, _, files in os.walk(out):
            for fn in files:
                path = os.path.join(root, fn)
                if os.path.getsize(path) < 1_000_000:
                    tar.add(path, arcname=os.path.relpath(path, out))
                else:
                    print(f"[{label}] non rapatrié (sur le volume, {dst}) : {fn}", flush=True)
    return buf.getvalue()


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=8.0, memory=32768,
              timeout=8 * 3600, volumes={"/data": vol})
def run(cmd: str, label: str = "") -> bytes:
    return _run(cmd, label)


@app.local_entrypoint()
def main(cmd: str):
    data = run.remote(cmd, f"{RUN_NAME} · GPU={GPU}")
    dest = os.environ.get("OUT_DIR", "out/modal")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")
    print(open(f"{dest}/log_modal.txt").read()[-4000:])
