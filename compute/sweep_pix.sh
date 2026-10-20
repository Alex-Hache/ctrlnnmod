#!/bin/bash
# M3a : 4 modèles × 2 tailles × 2 graines, un conteneur L4 par run, en parallèle.
# Usage : bash compute/sweep_pix.sh <dossier_local_des_résultats>
OUT=${1:-out/pix_sweep}; mkdir -p "$OUT"
for seed in 0 1; do for n in 256 1024; do for m in gru ren sren mamba2; do
  name=pix-$m-n$n-s$seed
  RUN_NAME=$name OUT_DIR=$OUT/$name MODAL_GPU=L4 MODAL_CPU=4 MODAL_MEM=16384 PYTHONUNBUFFERED=1 \
    timeout 10000 modal run compute/modal_job.py \
    --cmd "python experiments/pixel_wm/pwm.py --model $m --n $n --H 50 --seed $seed" > "$OUT/$name.log" 2>&1 &
  sleep 2
done; done; done
wait
echo "sweep fini"; ls "$OUT"/*/pix_*.json 2>/dev/null | wc -l
