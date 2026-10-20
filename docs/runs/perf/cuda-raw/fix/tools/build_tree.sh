#!/bin/bash
# Build one source tree like ~/tf108 and src108: dev CLI + tf-cuda-test (install), then tensorfold-native (native).
set -u
T=$1
cd $HOME/tf-pocket-cuda/$T
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
Z=~/pr144-review/zig/zig
for step in install native; do
  echo "START $step $(date -u +%FT%TZ)"
  s=$(date +%s.%N)
  nice -n 10 $Z build $step -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j8
  rc=$?
  e=$(date +%s.%N)
  echo "END $step $(date -u +%FT%TZ) rc=$rc wall=$(echo "$e - $s" | bc)s"
  echo $rc > $HOME/tf-pocket-cuda/build-$T-$step.rc
  [ $rc -eq 0 ] || exit $rc
done
