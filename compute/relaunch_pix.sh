#!/bin/bash
# Relance les runs M3a qui n'ont pas démarré (limite de création d'apps Modal) : un départ
# toutes les 30 s, et nouvel essai si la limite revient.
OUT=$1; shift
for name in "$@"; do
  m=$(echo $name | cut -d- -f2); n=$(echo $name | cut -d- -f3 | tr -d n); s=$(echo $name | cut -d- -f4 | tr -d s)
  for try in 1 2 3 4; do
    RUN_NAME=$name OUT_DIR=$OUT/$name MODAL_GPU=L4 MODAL_CPU=4 MODAL_MEM=16384 PYTHONUNBUFFERED=1 \
      timeout 10000 modal run compute/modal_job.py \
      --cmd "python experiments/pixel_wm/pwm.py --model $m --n $n --H 50 --seed $s" > "$OUT/$name.log" 2>&1 &
    pid=$!
    sleep 30
    if grep -aq "rate limit exceeded" "$OUT/$name.log" && ! kill -0 $pid 2>/dev/null; then
      echo "$name : limite, nouvel essai"; sleep 30; continue
    fi
    break
  done
done
wait
echo "relance finie"
