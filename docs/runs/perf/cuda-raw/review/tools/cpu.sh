#!/bin/bash
# CPU-only checks of the review tree, low priority: fatbins (srcR, srcG), zig fmt --check, zig build test (srcR).
set -u
R=$HOME/tf-pocket-cuda/review
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
Z=~/pr144-review/zig/zig
for t in srcR srcG; do (cd $R/$t && nice -n 19 $Z build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j4 > $R/raw/fatbins-$t.log 2>&1; echo "fatbins $t rc=$?"); done
(cd $R/srcR && $Z fmt --check zig/src/families/nemotron zig/tests/cuda zig/src/cuda; echo "fmt rc=$?")
(cd $R/srcR && echo "START test $(date -u +%FT%TZ)" && nice -n 19 $Z build test -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j4 --summary all > $R/raw/test-srcR.log 2>&1; echo "END test $(date -u +%FT%TZ) rc=$?")
echo CPUDONE
