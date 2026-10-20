#!/bin/bash
# GPU ms a serial and a 1-row window graph round (dev CLI `rounds`), before and after on A, T, E.
# Folders rotated, builds ABBA (alternating each rep), GAP s idle before each rep. usage: rounds2.sh TAG REPS GAP_S
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
check_clis
IDS=$(cat $H/tools/p1-off.ids)
TAG=$1; N=$2; GAP=$3
O=$H/raw/v/rounds-$TAG; mkdir -p $O
for rep in $(seq 1 $N); do
  sleep $GAP
  case $((rep % 3)) in 1) ORDER="A T E";; 2) ORDER="T E A";; 0) ORDER="E A T";; esac
  case $((rep % 2)) in 1) BO="before after";; 0) BO="after before";; esac
  for m in $ORDER; do for b in $BO; do
    T0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    stop_check
    wait_free
    ${CLI[$b]} rounds ${M[$m]} --tokens $IDS --max-tokens 400 > $O/$m-$b-$rep.log 2>&1
    echo "$(date -u +%T) $m $b $rep exit=$? temp0=$T0 $(grep RESULT $O/$m-$b-$rep.log | sed 's/RESULT //' | tr '\n' ' ')"
  done; done
done
echo "$(date -u +%T) done"
