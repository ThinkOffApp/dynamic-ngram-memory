#!/bin/bash
# Clean 0efbd447 build on gb10-a: dev CLI, tf-cuda-test and tensorfold-native (CPU compile only).
set -u
cd $HOME/tf-pocket-cuda/src108
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
echo "START $(date -u +%FT%TZ)"
s=$(date +%s.%N)
nice -n 10 ~/pr144-review/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j6
rc=$?
e=$(date +%s.%N)
echo "END $(date -u +%FT%TZ) rc=$rc wall=$(echo "$e - $s" | bc)s"
echo $rc > $HOME/tf-pocket-cuda/build108.rc
