#!/bin/bash
# CPU builds of tensorfold-native for srcC (adec42df) and srcM (adec42df, recapture only for a pocket made now).
set -u
H=~/tf-pocket-cuda
export ZIG_GLOBAL_CACHE_DIR=$H/.zig-global
for T in srcC srcM; do
  cd $H/$T
  echo "START $T native $(date -u +%FT%TZ)"
  nice -n 15 ~/pr144-review/zig/zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j8 > $H/live/build-$T.log 2>&1
  echo "END $T native rc=$? $(date -u +%FT%TZ)"
  sha256sum zig-out/native/bin/tensorfold-native
done
