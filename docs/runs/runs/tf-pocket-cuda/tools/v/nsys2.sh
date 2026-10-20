#!/bin/bash
# nsys traces of the dev CLI's decode (kernels inside graphs traced node by node), before or after build.
# usage: nsys2.sh TAG "after:T:nd after:T:d after:E:nd before:T:nd ..."
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
take_lock
check_clis
IDS=$(cat $H/tools/p1-off.ids)
TAG=$1; LIST=$2
O=$H/raw/v/nsys-$TAG; mkdir -p $O
for x in $LIST; do
  IFS=: read b m d <<< "$x"
  FL="--no-drafts"; [ $d = d ] && FL=""
  sleep 30
  stop_check
  wait_free
  nsys profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/$b-$m-$d ${CLI[$b]} run ${M[$m]} --tokens $IDS --max-tokens 256 $FL --report $O/$b-$m-$d.report.json > $O/$b-$m-$d.log 2>&1
  echo "$(date -u +%T) $b $m $d profile exit=$? $(grep -E '^tokens' $O/$b-$m-$d.log)"
  nsys export --type sqlite --force-overwrite=true --output $O/$b-$m-$d.sqlite $O/$b-$m-$d.nsys-rep > /dev/null 2>&1
  python3 -I $H/tools/v/nsys_kern2.py $O/$b-$m-$d.sqlite $O/$b-$m-$d.kern.json > $O/$b-$m-$d.kern.txt 2>&1
  echo "$(date -u +%T) $b $m $d exported $(grep -c . $O/$b-$m-$d.kern.txt) lines"
done
echo "$(date -u +%T) done"
