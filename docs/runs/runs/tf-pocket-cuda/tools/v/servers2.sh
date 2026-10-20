#!/bin/bash
# Servers one at a time on 18190 (srv2.py): ORDER of build-folder pairs (bT = before on T), GAP s idle before each.
# usage: servers2.sh SERIES GAP "bT aT aA bA bE aE" ACTIONS
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
SERIES=$1; GAP=$2; ORDER=$3; ACTIONS=$4
declare -A BN=([b]=before [a]=after)
i=0
for x in $ORDER; do
  i=$((i+1)); b=${BN[${x:0:1}]}; m=${x:1:1}
  sleep $GAP
  stop_check
  wait_free
  echo "$(date -u +%T) server $SERIES-$i-$x ($b on $m) smi: $(smi)"
  python3 -I $H/tools/v/srv2.py $SERIES $i-$x $b $m $ACTIONS
  echo "$(date -u +%T) server $SERIES-$i-$x rc=$?"
done
echo "$(date -u +%T) done"
