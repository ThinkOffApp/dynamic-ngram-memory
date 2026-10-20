#!/bin/bash
# Dev CLI decode bursts, before and after on A, T, E: p1 (36 ids), 256 greedy tokens, drafts off and on.
# Each burst: folders rotated, builds in ABBA order (alternating each burst), GAP s idle before each burst.
# usage: clib.sh TAG BURSTS GAP_S
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
check_clis
IDS=$(cat $H/tools/p1-off.ids)
TAG=$1; N=$2; GAP=$3
O=$H/raw/v/cli-$TAG; mkdir -p $O
hashes > $O/hash-before.txt
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  case $((rep % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for m in $ORDER; do for b in $BO; do for d in nd d; do
    FL="--no-drafts"; [ $d = d ] && FL=""
    stop_check
    wait_free
    echo "$(date -u +%T) start $m $b $d $rep smi: $(smi)" >> $O/smi.log
    ${CLI[$b]} run ${M[$m]} --tokens $IDS --max-tokens 256 $FL --report $O/$m-$b-$d-$rep.json > $O/$m-$b-$d-$rep.log 2>&1
    echo "$(date -u +%T) $m $b $d $rep exit=$? $(grep -E '^tokens' $O/$m-$b-$d-$rep.log)"
  done; done; done
done
hashes > $O/hash-after.txt
if cmp -s $O/hash-before.txt $O/hash-after.txt; then echo "$(date -u +%T) folders unchanged ($(wc -l < $O/hash-after.txt) files)"; else echo "$(date -u +%T) FOLDERS CHANGED"; fi
echo "$(date -u +%T) done"
