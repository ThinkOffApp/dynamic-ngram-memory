#!/bin/bash
# Prompt cost: a 4,096-token prompt (p1's ids repeated), 2 tokens out, drafts off; A/T/E rotated, idle gaps.
# usage: prefill.sh TAG REPS GAP_S
set -u
# one GPU job at a time: the lock, and no other engine process
exec 9>~/tf-pocket-cuda/GPU.lock
flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "nsys profile" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another engine process: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test)"; sleep 30; done; }
H=~/tf-pocket-cuda
CLI=$H/src108/zig-out/bin/tensorfold
declare -A M=([A]=~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8 [T]=~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main [E]=$H/models/A-empty64)
TAG=$1; N=$2; GAP=$3
O=$H/raw/prefill-$TAG; mkdir -p $O
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  for m in $ORDER; do
    wait_free; $CLI run ${M[$m]} --tokens-file $H/tools/p4096.ids --max-tokens 2 --no-drafts --report $O/$m-$rep.json > $O/$m-$rep.log 2>&1
    echo "$(date -u +%T) $m $rep exit=$? $(grep -E '^tokens' $O/$m-$rep.log)"
  done
done
echo "$(date -u +%T) done"
