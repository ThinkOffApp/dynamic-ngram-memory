#!/bin/bash
# Copy run6/model to the taught copy; stat and hash source and copy around it.
set -e
E=$HOME/tf-promote/e1-20261019
SRC=$HOME/tf-promote/run6/model
DST=$E/model-taught
V=$E/validation
date --iso-8601=ns > $V/taught-copy.start
grep "notes promotion: night" $HOME/tf-promote/run6/server.log | tail -2 > $V/taught-copy.run6-nights-before.txt
cp $HOME/tf-promote/run6/nights-day1.snap.json.promote $V/taught-copy.run6-promote-before.json
(cd $SRC && stat -c "%n %s %y" $(ls -A) ) > $V/taught-copy.src-stat-before.txt
mkdir -p $DST
cp -a $SRC/. $DST/
(cd $SRC && stat -c "%n %s %y" $(ls -A) ) > $V/taught-copy.src-stat-after-copy.txt
(cd $DST && sha256sum $(ls -A) ) > $V/taught-copy.dst-sha256.txt
(cd $SRC && sha256sum $(ls -A) ) > $V/taught-copy.src-sha256.txt
(cd $SRC && stat -c "%n %s %y" $(ls -A) ) > $V/taught-copy.src-stat-after-hash.txt
grep "notes promotion: night" $HOME/tf-promote/run6/server.log | tail -2 > $V/taught-copy.run6-nights-after.txt
date --iso-8601=ns > $V/taught-copy.end
echo done > $V/taught-copy.done
