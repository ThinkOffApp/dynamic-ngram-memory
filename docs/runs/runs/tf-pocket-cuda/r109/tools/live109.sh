#!/bin/bash
# 1.0.9 receipt, live learning (phase 3 of live/run_live3.sh + run_restart3.sh), run under GPU.lock (chain109b.sh):
# before on its own copy, after on its own copy checked against before at every checkpoint, then each folder restarted.
set -u
H=~/tf-pocket-cuda
R=$H/r109
mkdir -p $R/live $R/models
python3 -I $R/tools/live109.py before; rc=$?; echo "$(date -u +%T) live before rc=$rc"; [ $rc -eq 0 ] || exit $rc
python3 -I $R/tools/live109.py after before; rc=$?; echo "$(date -u +%T) live after rc=$rc"; [ $rc -eq 0 ] || exit $rc
python3 -I $R/tools/live109.py restart before before; echo "$(date -u +%T) restart before rc=$?"
python3 -I $R/tools/live109.py restart after after; echo "$(date -u +%T) restart after rc=$?"
(cd $R/models && for d in p3-before p3-after; do (cd $d && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $R/live/final-$d.sha256; done)
du -sh $R/models/p3-before $R/models/p3-after
echo "$(date -u +%T) live done"
