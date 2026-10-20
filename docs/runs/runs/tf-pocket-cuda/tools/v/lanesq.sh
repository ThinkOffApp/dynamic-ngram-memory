#!/bin/bash
# Fact questions through the dev CLI's lane core (`tensorfold lanes`), full token ids, before and after on A, T, E:
# thinking off (all 134 solo, the 40 together; 160 tokens) and on (the 40; 1500 tokens); solo drafted, solo --no-drafts,
# together drafted (the lane core takes at most 63 sequences at once).
# usage: lanesq.sh TAG "T A E"
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
check_clis
TAG=$1; ORDER=$2
O=$H/raw/v/lanes-$TAG; mkdir -p $O
F=$H/tools/v/facts
hashes > $O/hash-before.txt
echo "$(date -u +%T) hashed $(wc -l < $O/hash-before.txt) files"
k=0
for m in $ORDER; do
  k=$((k+1)); case $((k % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for b in $BO; do
    for set in off-all:160:solo-d off-all:160:solo-nd off-40:160:together-d on-40:1500:solo-d on-40:1500:solo-nd on-40:1500:together-d; do
      IFS=: read p mt mode <<< "$set"
      for once in 1; do
        FL=""; case $mode in solo-d) FL="--solo";; solo-nd) FL="--solo --no-drafts";; together-d) FL="";; esac
        wait_free
        s=$(date +%s)
        ${CLI[$b]} lanes ${M[$m]} $F/facts-$p.json $FL --max-tokens $mt --report $O/$m-$b-$p-$mode.json > $O/$m-$b-$p-$mode.log 2>&1
        rc=$?
        echo "$(date -u +%T) $m $b $p $mode exit=$rc $(( $(date +%s) - s ))s temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) $(grep -E '^(solo|together):' $O/$m-$b-$p-$mode.log)"
      done
    done
  done
done
hashes > $O/hash-after.txt
if cmp -s $O/hash-before.txt $O/hash-after.txt; then echo "$(date -u +%T) folders unchanged ($(wc -l < $O/hash-after.txt) files)"; else echo "$(date -u +%T) FOLDERS CHANGED"; fi
echo "$(date -u +%T) done"
