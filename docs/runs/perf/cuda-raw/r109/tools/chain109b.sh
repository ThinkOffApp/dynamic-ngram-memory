#!/bin/bash
# 1.0.9 receipt, GPU stage b under GPU.lock: the server series (servers109.sh s1, mirrored build order per folder,
# 120 s idle before each server), then live learning (live109.sh). Waits for the lock and for other GPU processes.
set -u
H=$HOME/tf-pocket-cuda
R=$H/r109
O=$R/raw
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null || pgrep -f "^python3 [^ ]*dedup_identical_copies" >/dev/null; }
exec 9>$H/GPU.lock
while true; do
  while busy; do sleep 30; done
  flock -w 86400 9 || { echo "$(date -u +%T) no lock in 24 h"; exit 3; }
  if busy; then flock -u 9; echo "$(date -u +%T) lock taken but another GPU process runs: released"; sleep 30; continue; fi
  break
done
echo "$(date -u +%T) GPU.lock taken; $(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader)"
bash $R/tools/servers109.sh s1 120 "bT aT aA bA bE aE" openai,openai256,off,conc,conc14 > $O/servers-s1.out 2>&1
echo "$(date -u +%T) servers rc=$? $(tail -1 $O/servers-s1.out)"
bash $R/tools/live109.sh > $O/live.out 2>&1
echo "$(date -u +%T) live rc=$? $(tail -1 $O/live.out)"
echo "$(date -u +%T) chain109b done"
