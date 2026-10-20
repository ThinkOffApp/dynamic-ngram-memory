#!/bin/bash
# tf-cuda-test window-profile (graph ms and eager ms by kernel class) before and after on the given folders.
# usage: wp2.sh TAG GAP_S "before:T after:T ..."
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
TAG=$1; GAP=$2; LIST=$3
O=$H/raw/v/wp-$TAG; mkdir -p $O
for x in $LIST; do
  IFS=: read b m <<< "$x"
  sleep $GAP
  stop_check
  wait_free
  ${TEST[$b]} window-profile ${M[$m]} 1,4,16 $H/tools/p1-off.ids > $O/$b-$m.log 2>&1
  echo "$(date -u +%T) $b $m exit=$? $(grep -c WINDOW $O/$b-$m.log) windows"
done
echo "$(date -u +%T) done"
