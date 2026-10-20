#!/bin/bash
# TensorFold v1.0.11 (ffa85fd5) on gb10-a for the 1.0.11 Living Weights seeds: same flags as build108.sh.
set -u
H=$HOME/tf-pocket-cuda
if [ ! -d $H/src111/.git ]; then git clone -q --depth 1 --branch v1.0.11 https://github.com/ashhart/TensorFold.git $H/src111 || { echo clone-failed; echo 90 > $H/build111.rc; exit 90; }; fi
cd $H/src111
echo "HEAD $(git rev-parse HEAD)"
export ZIG_GLOBAL_CACHE_DIR=$H/.zig-global
echo "START $(date -u +%FT%TZ)"
s=$(date +%s.%N)
nice -n 10 ~/pr144-review/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.11 -j6
rc=$?
e=$(date +%s.%N)
echo "END $(date -u +%FT%TZ) rc=$rc wall=$(echo "$e - $s" | bc)s"
echo $rc > $H/build111.rc
