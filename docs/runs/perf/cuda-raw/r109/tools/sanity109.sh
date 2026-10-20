#!/bin/bash
# 1.0.9 receipt, dev CLI: A, T, E x before (c36b536d) / after (76fe1de4) x drafts off/on, 256 greedy tokens from p1 (36 ids),
# N reps, folders rotated each rep, build order alternated each rep (before first on odd reps); then the 4,096-token prompt
# (--max-tokens 2 --no-drafts), N reps alike. Folders hashed before and after. Run under GPU.lock (chain109a.sh).
# usage: sanity109.sh TAG REPS GAP_S
set -u
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null || pgrep -f "^python3 [^ ]*dedup_identical_copies" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another engine process"; sleep 30; done; }
H=~/tf-pocket-cuda
R=$H/r109
declare -A B=([before]=$R/before/zig-out/bin/tensorfold [after]=$R/after/zig-out/bin/tensorfold)
declare -A M=([A]=~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8 [T]=~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main [E]=$H/models/A-empty64)
IDS=$(cat $H/tools/p1-off.ids)
TAG=$1; N=$2; GAP=$3
O=$R/raw/sanity-$TAG; mkdir -p $O
for b in before after; do
  want=$(grep " zig-out/bin/tensorfold$" $R/raw/bin-$b.sha256 | cut -c1-64); got=$(sha256sum < ${B[$b]} | cut -c1-64)
  [ "$want" = "$got" ] || { echo "refusing: ${B[$b]} sha256 $got, recorded $want"; exit 3; }
  echo "$b cli sha256 $got"
done
hashes() { for m in A T E; do (cd ${M[$m]} && for f in config.json *.safetensors; do echo "$m $f $(sha256sum < "$f" | cut -c1-64)"; done); done; }
hashes > $O/hash-before.txt
echo "$(date -u +%T) hashed $(wc -l < $O/hash-before.txt) files"
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  case $((rep % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for m in $ORDER; do for b in $BO; do for d in nd d; do
    FL="--no-drafts"; [ $d = d ] && FL=""
    wait_free; ${B[$b]} run ${M[$m]} --tokens $IDS --max-tokens 256 $FL --report $O/$m-$b-$d-$rep.json > $O/$m-$b-$d-$rep.log 2>&1
    echo "$(date -u +%T) $m $b $d $rep exit=$? temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) $(grep -E "^tokens" $O/$m-$b-$d-$rep.log)"
  done; done; done
done
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  case $((rep % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for m in $ORDER; do for b in $BO; do
    wait_free; ${B[$b]} run ${M[$m]} --tokens-file $H/tools/p4096.ids --max-tokens 2 --no-drafts --report $O/p4096-$m-$b-$rep.json > $O/p4096-$m-$b-$rep.log 2>&1
    echo "$(date -u +%T) p4096 $m $b $rep exit=$? temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) $(grep -E "^tokens" $O/p4096-$m-$b-$rep.log)"
  done; done
done
hashes > $O/hash-after.txt
if cmp -s $O/hash-before.txt $O/hash-after.txt; then echo "$(date -u +%T) folders unchanged ($(wc -l < $O/hash-after.txt) files)"; else echo "$(date -u +%T) FOLDERS CHANGED"; fi
echo "$(date -u +%T) done"
