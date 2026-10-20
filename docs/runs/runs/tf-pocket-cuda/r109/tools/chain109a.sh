#!/bin/bash
# 1.0.9 receipt, GPU stage a under GPU.lock: tf-cuda-test pocket on after (and on before, which has no such command),
# then the dev CLI sanity (sanity109.sh c1 4 45). Waits for the lock and for any other engine, test, nsys or dedup process.
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
job() { local log=$1; shift; echo "START $(date -u +%FT%TZ) temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) cmd: $*" > $log; "$@" >> $log 2>&1; local rc=$?; echo "END $(date -u +%FT%TZ) rc=$rc" >> $log; echo "$(date -u +%T) $(basename $log) rc=$rc"; }
job $O/pocket-after.log $R/after/zig-out/bin/tf-cuda-test pocket 30
job $O/pocket-before.log $R/before/zig-out/bin/tf-cuda-test pocket 30
bash $R/tools/sanity109.sh c1 4 45
echo "$(date -u +%T) chain109a done"
