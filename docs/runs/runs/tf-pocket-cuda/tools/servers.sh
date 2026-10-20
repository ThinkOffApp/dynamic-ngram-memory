#!/bin/bash
# 1.0.8 servers one at a time on 18190: ORDER of A/T/E, GAP s idle before each (temperature logged).
# usage: servers.sh TAG GAP "A T E E T A" ACTIONS
set -u
# one GPU job at a time: the lock, and no other engine process
exec 9>~/tf-pocket-cuda/GPU.lock
flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "nsys profile" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another engine process: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test)"; sleep 30; done; }
H=~/tf-pocket-cuda
TAG=$1; GAP=$2; ORDER=$3; ACTIONS=$4
i=0
for m in $ORDER; do
  i=$((i+1))
  sleep $GAP
  echo "$(date -u +%T) server $TAG-$i-$m temp $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader)"
  wait_free; python3 -I $H/tools/srv.py $TAG-$i-$m $m $ACTIONS
done
echo "$(date -u +%T) done"
