#!/bin/bash
# nsys traces of the dev CLI's decode (kernels inside graphs traced node by node).
# usage: nsys.sh MODEL(A|T|E) MODE(nd|d) TAG
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
m=$1; d=$2; TAG=$3
O=$H/raw/nsys; mkdir -p $O
FL="--no-drafts"; [ $d = d ] && FL=""
wait_free; nsys profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/$TAG-$m-$d $CLI run ${M[$m]} --tokens $IDS --max-tokens 256 $FL --report $O/$TAG-$m-$d.report.json > $O/$TAG-$m-$d.log 2>&1
echo "profile exit=$? $(grep -oE 'tokens [0-9]+ sha [0-9a-f]+ .* ms/token rounds [0-9]+ accepted [0-9]+' $O/$TAG-$m-$d.log)"
nsys export --type sqlite --force-overwrite=true --output $O/$TAG-$m-$d.sqlite $O/$TAG-$m-$d.nsys-rep > /dev/null 2>&1
python3 -I $H/tools/nsys_kern.py $O/$TAG-$m-$d.sqlite $O/$TAG-$m-$d.kern.json
