#!/bin/bash
# Phase 3 under GPU.lock: before (0efbd447), then tip (adec42df) and the recapture mutant, each checked against before.
set -u
H=~/tf-pocket-cuda
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null; }
exec 9>$H/GPU.lock
while true; do
  while busy; do echo "$(date -u +%T) waiting: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test)"; sleep 30; done
  flock -w 86400 9 || { echo "$(date -u +%T) no lock in 24 h"; exit 3; }
  if busy; then flock -u 9; echo "$(date -u +%T) lock taken but another GPU process runs: released"; sleep 30; continue; fi
  break
done
echo "$(date -u +%T) GPU.lock taken; $(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader)"
python3 -I $H/live/live3.py before
rc=$?
echo "$(date -u +%T) before rc=$rc"
[ $rc -eq 0 ] || exit $rc
python3 -I $H/live/live3.py tip before
rc=$?
echo "$(date -u +%T) tip rc=$rc"
[ $rc -eq 0 ] || exit $rc
python3 -I $H/live/live3.py mutant before
rc=$?
echo "$(date -u +%T) mutant rc=$rc"
exit $rc
