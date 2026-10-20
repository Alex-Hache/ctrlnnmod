"""Rapatrie les petits fichiers de résultats du volume Modal (JSON, PNG, logs) sans passer par
le stockage objet : le retour d'une fonction de moins de ~2 Mo arrive directement au client.

    modal run compute/fetch_volume.py --pattern 'pix-mamba2*' --dest out/fetched

Utile quand le client local d'un run a été coupé : le conteneur finit seul et copie ses
résultats dans /data/runs/<RUN_NAME>/.
"""
import io
import os
import tarfile

import modal

app = modal.App("ctrlnnmod-fetch")
vol = modal.Volume.from_name("ctrlnnmod-data", create_if_missing=True)


@app.function(volumes={"/data": vol}, timeout=600)
def pack(pattern: str, exts: str = ".json,.png,.txt") -> bytes:
    import fnmatch
    vol.reload()
    root = "/data/runs"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for run in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if not fnmatch.fnmatch(run, pattern):
                continue
            for fn in os.listdir(f"{root}/{run}"):
                p = f"{root}/{run}/{fn}"
                if fn.endswith(tuple(exts.split(","))) and os.path.getsize(p) < 500_000:
                    tar.add(p, arcname=f"{run}/{fn}")
    return buf.getvalue()


@app.local_entrypoint()
def main(pattern: str = "*", dest: str = "out/fetched"):
    data = pack.remote(pattern)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        names = tar.getnames()
        tar.extractall(dest, filter="data")
    print(len(names), "fichiers →", dest)
    for n in names:
        print(" ", n)
