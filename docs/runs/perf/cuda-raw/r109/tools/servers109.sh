#!/bin/bash
# 1.0.9 receipt: servers one at a time on 18190 (srv109.py), ORDER of build-folder pairs (bT = before on T), GAP s idle before each.
# Run under GPU.lock (chain109b.sh); waits for any other engine, test, nsys or dedup process. usage: servers109.sh SERIES GAP "ORDER" ACTIONS
set -u
H=~/tf-pocket-cuda
R=$H/r109
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null || pgrep -f "^[^ ]*python[0-9.]* [^ ]*dedup_identical_copies" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another GPU or dedup process"; sleep 30; done; }
smi() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader; }
SERIES=$1; GAP=$2; ORDER=$3; ACTIONS=$4
declare -A BN=([b]=before [a]=after)
i=0
for x in $ORDER; do
  i=$((i+1)); b=${BN[${x:0:1}]}; m=${x:1:1}
  sleep $GAP
  [ -e $R/STOP ] && { echo "$(date -u +%T) STOP file present: stopping"; exit 5; }
  wait_free
  echo "$(date -u +%T) server $SERIES-$i-$x ($b on $m) smi: $(smi)"
  python3 -I $R/tools/srv109.py $SERIES $i-$x $b $m $ACTIONS
  echo "$(date -u +%T) server $SERIES-$i-$x rc=$?"
done
echo "$(date -u +%T) done"
