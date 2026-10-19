"""Planche de quelques images du jeu de données (contrôle visuel du rendu)."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

z = np.load("/data/franka_pix.npz")
I = z["Itr"]
fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
for j in range(6):
    for k in range(2):
        ax[k, j].imshow(I[j, 2 * k + 1, 0].transpose(1, 2, 0))
        ax[k, j].axis("off")
fig.suptitle(f"Franka, 64×64 — {I.shape[0]} trajectoires × {I.shape[1]} départs × 2 images")
fig.tight_layout()
fig.savefig(os.path.join(os.environ.get("OUT_DIR", "."), "frames.png"), dpi=100)
print("frames.png écrit ; moyenne pixels", I.mean(), "écart-type entre images", I.std(0).mean())
