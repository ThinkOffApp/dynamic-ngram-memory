#!/bin/bash
# CPU-only compile check of srcB (install step: dev CLI + tf-cuda-test), low priority.
set -u
cd $HOME/tf-pocket-cuda/srcB
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
echo "START install $(date -u +%FT%TZ)"
nice -n 19 ~/pr144-review/zig/zig build install -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j4
rc=$?
echo "END install $(date -u +%FT%TZ) rc=$rc"
