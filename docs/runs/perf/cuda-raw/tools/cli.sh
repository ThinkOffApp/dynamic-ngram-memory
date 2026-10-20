#!/bin/bash
# Dev CLI (clean 0efbd447 build, same fatbins as the 1.0.8 server) on gb10-a: decode bursts over A, T, E.
# usage: cli.sh TAG BURSTS GAP_S
set -u
# one GPU job at a time: the lock, and no other engine process
exec 9>~/tf-pocket-cuda/GPU.lock
flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "nsys profile" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another engine process: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test)"; sleep 30; done; }
H=~/tf-pocket-cuda
CLI=$H/src108/zig-out/bin/tensorfold
IDS=$(cat $H/tools/p1-off.ids)
declare -A M=([A]=~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8 [T]=~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main [E]=$H/models/A-empty64)
TAG=$1; N=$2; GAP=$3
O=$H/raw/cli-$TAG; mkdir -p $O
smi() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader; }
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  for m in $ORDER; do
    for d in nd d; do
      FL="--no-drafts"; [ $d = d ] && FL=""
      echo "$(date -u +%T) start $m $d $rep smi: $(smi)" >> $O/smi.log
      wait_free; $CLI run ${M[$m]} --tokens $IDS --max-tokens 256 $FL --report $O/$m-$d-$rep.json > $O/$m-$d-$rep.log 2>&1
      rc=$?
      echo "$(date -u +%T) $m $d $rep exit=$rc $(grep -oE 'tokens [0-9]+ sha [0-9a-f]+ prefill [0-9.]+s decode [0-9.]+s [0-9.]+ ms/token rounds [0-9]+ accepted [0-9]+' $O/$m-$d-$rep.log)"
    done
  done
done
echo "$(date -u +%T) done"
