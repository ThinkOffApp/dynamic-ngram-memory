#!/bin/bash
# Same tokens before and after: dev CLI on A, T, E with the 0efbd447 build and the pocket-cuda build, drafts off and
# on, then the 4,096-token prompt. Folders hashed before and after. usage: sanity.sh TAG REPS GAP_S
set -u
exec 9>~/tf-pocket-cuda/GPU.lock
flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null || pgrep -f "^python3 [^ ]*dedup_identical_copies" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another engine process"; sleep 30; done; }
H=~/tf-pocket-cuda
declare -A B=([before]=$H/before/zig-out/bin/tensorfold [after]=$H/src/zig-out/bin/tensorfold)
declare -A M=([A]=~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8 [T]=~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main [E]=$H/models/A-empty64)
IDS=$(cat $H/tools/p1-off.ids)
TAG=$1; N=$2; GAP=$3
O=$H/raw/sanity-$TAG; mkdir -p $O
hashes() { for m in A T E; do (cd ${M[$m]} && for f in config.json *.safetensors; do echo "$m $f $(sha256sum < "$f" | cut -c1-64)"; done); done; }
hashes > $O/hash-before.txt
echo "$(date -u +%T) hashed $(wc -l < $O/hash-before.txt) files"
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for m in A T E; do for b in $BO; do for d in nd d; do
    FL="--no-drafts"; [ $d = d ] && FL=""
    wait_free; ${B[$b]} run ${M[$m]} --tokens $IDS --max-tokens 256 $FL --report $O/$m-$b-$d-$rep.json > $O/$m-$b-$d-$rep.log 2>&1
    echo "$(date -u +%T) $m $b $d $rep exit=$? temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) $(grep -E "^tokens" $O/$m-$b-$d-$rep.log)"
  done; done; done
done
for m in A T E; do for b in before after; do
  wait_free; ${B[$b]} run ${M[$m]} --tokens-file $H/tools/p4096.ids --max-tokens 2 --no-drafts --report $O/p4096-$m-$b.json > $O/p4096-$m-$b.log 2>&1
  echo "$(date -u +%T) p4096 $m $b exit=$? $(grep -E "^tokens" $O/p4096-$m-$b.log)"
done; done
hashes > $O/hash-after.txt
if cmp -s $O/hash-before.txt $O/hash-after.txt; then echo "$(date -u +%T) folders unchanged ($(wc -l < $O/hash-after.txt) files)"; else echo "$(date -u +%T) FOLDERS CHANGED"; fi
echo "$(date -u +%T) done"
