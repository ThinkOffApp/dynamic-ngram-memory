#!/bin/bash
# Prompt cost: 4,096 tokens (p1 repeated), 2 out, drafts off; before and after on A, T, E, folders rotated, builds ABBA.
# usage: prefill2.sh TAG REPS GAP_S
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
check_clis
TAG=$1; N=$2; GAP=$3
O=$H/raw/v/prefill-$TAG; mkdir -p $O
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  case $((rep % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for m in $ORDER; do for b in $BO; do
    stop_check
    wait_free
    ${CLI[$b]} run ${M[$m]} --tokens-file $H/tools/p4096.ids --max-tokens 2 --no-drafts --report $O/$m-$b-$rep.json > $O/$m-$b-$rep.log 2>&1
    echo "$(date -u +%T) $m $b $rep exit=$? $(grep -E '^tokens' $O/$m-$b-$rep.log)"
  done; done
done
echo "$(date -u +%T) done"
