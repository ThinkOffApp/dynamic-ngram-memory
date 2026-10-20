#!/bin/bash
# zig build test (host unit tests) in one tree, with the build flags of build_tree.sh; summary of every step.
set -u
T=$1
cd $HOME/tf-pocket-cuda/$T
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
echo "START test $(date -u +%FT%TZ)"
nice -n 10 ~/pr144-review/zig/zig build test -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j8 --summary all
rc=$?
echo "END test $(date -u +%FT%TZ) rc=$rc"
