#!/bin/bash
# Lance des runs Modal espacés de 30 s (limite de création d'apps), un par ligne du fichier :
#   <RUN_NAME> <commande...>
# Usage : bash compute/launch_spaced.sh jobs.txt <dossier_local_des_résultats>
JOBS=$1; OUT=$2; mkdir -p "$OUT"
while read -r name cmd; do
  [ -z "$name" ] && continue
  for try in 1 2 3 4; do
    RUN_NAME=$name OUT_DIR=$OUT/$name MODAL_GPU=${MODAL_GPU:-L4} MODAL_CPU=${MODAL_CPU:-4} \
      MODAL_MEM=${MODAL_MEM:-16384} PYTHONUNBUFFERED=1 timeout 10000 \
      modal run compute/modal_job.py --cmd "$cmd" > "$OUT/$name.log" 2>&1 < /dev/null &
    pid=$!
    sleep 30
    if grep -aq "rate limit exceeded" "$OUT/$name.log" && ! kill -0 $pid 2>/dev/null; then
      echo "$name : limite, nouvel essai"; sleep 30; continue
    fi
    break
  done
done < "$JOBS"
wait
echo "tous les runs sont finis"
