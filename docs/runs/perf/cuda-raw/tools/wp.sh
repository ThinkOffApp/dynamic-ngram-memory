#!/bin/bash
# tf-cuda-test window-profile (graph ms and eager ms by kernel class, forked and one stream) for A/T/E.
# usage: wp.sh TAG GAP_S "A T E"
set -u
# one GPU job at a time: the lock, and no other engine process
exec 9>~/tf-pocket-cuda/GPU.lock
flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "nsys profile" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another engine process: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test)"; sleep 30; done; }
H=~/tf-pocket-cuda
TEST=$H/src108/zig-out/bin/tf-cuda-test
declare -A M=([A]=~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8 [T]=~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main [E]=$H/models/A-empty64)
TAG=$1; GAP=$2; ORDER=$3
O=$H/raw/wp-$TAG; mkdir -p $O
for m in $ORDER; do
  sleep $GAP
  wait_free; $TEST window-profile ${M[$m]} 1,4,16 $H/tools/p1-off.ids > $O/$m.log 2>&1
  echo "$(date -u +%T) $m exit=$? $(grep -c WINDOW $O/$m.log) windows"
done
echo "$(date -u +%T) done"
