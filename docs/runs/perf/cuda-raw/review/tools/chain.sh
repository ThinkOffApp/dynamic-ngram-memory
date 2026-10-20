#!/bin/bash
# Review GPU jobs, one at a time under GPU.lock (waits for it and for any other engine, test, nsys or dedup process).
set -u
H=$HOME/tf-pocket-cuda
R=$H/review
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
cd $R
job $O/pocket-srcR.log srcR/zig-out/bin/tf-cuda-test pocket 30
job $O/pocket-srcF.log srcF/zig-out/bin/tf-cuda-test pocket 0
job $O/pocket-srcG.log srcG/zig-out/bin/tf-cuda-test pocket 0
bash $R/tools/sanity.sh r1 2 30
echo "$(date -u +%T) chain done"
