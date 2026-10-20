#!/bin/bash
# Build review trees like build_tree.sh (install, then native for srcR), low priority: build.sh TREE [native]
set -u
R=$HOME/tf-pocket-cuda/review
T=$1
cd $R/$T
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
Z=~/pr144-review/zig/zig
steps="install"; [ "${2:-}" = native ] && steps="install native"
for step in $steps; do
  echo "START $T $step $(date -u +%FT%TZ)"
  nice -n 19 $Z build $step -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j4
  rc=$?
  echo "END $T $step $(date -u +%FT%TZ) rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
