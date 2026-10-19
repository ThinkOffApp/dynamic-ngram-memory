#!/bin/bash
# Copy SNAP (dereferenced) to the untaught copy; check every blob against its name, and the copy against the blobs.
set -e
E=$HOME/tf-promote/e1-20261019
SNAP=$HOME/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8
DST=$E/model-untaught
V=$E/validation
date --iso-8601=ns > $V/untaught-copy.start
mkdir -p $DST
cp -RL $SNAP/. $DST/
: > $V/untaught-blobcheck.txt
for f in $(ls -A $SNAP); do
  blob=$(basename $(readlink $SNAP/$f))
  if [ ${#blob} -eq 64 ]; then got=$(sha256sum < $SNAP/$f | cut -d" " -f1); kind=sha256
  else got=$(git hash-object $SNAP/$f); kind=git-sha1; fi
  copy=$( [ $kind = sha256 ] && sha256sum < $DST/$f | cut -d" " -f1 || git hash-object $DST/$f )
  ok=$( [ "$got" = "$blob" ] && [ "$copy" = "$blob" ] && echo OK || echo MISMATCH )
  echo "$ok $kind $f blob=$blob src=$got copy=$copy" >> $V/untaught-blobcheck.txt
done
(cd $DST && sha256sum $(ls -A)) > $V/untaught-copy.dst-sha256.txt
date --iso-8601=ns > $V/untaught-copy.end
echo done > $V/untaught-copy.done
