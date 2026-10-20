#!/bin/bash
# The calibration: srcN (adec42df, no recapture at all) built on the CPU, then its live run under GPU.lock.
set -u
H=~/tf-pocket-cuda
export ZIG_GLOBAL_CACHE_DIR=$H/.zig-global
cd $H/srcN && grep -n "MUTANT" zig/src/families/nemotron/cuda_slide.zig
nice -n 15 ~/pr144-review/zig/zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j8 > $H/live/build-srcN.log 2>&1
echo "srcN native rc=$?"; sha256sum zig-out/native/bin/tensorfold-native
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null; }
exec 9>$H/GPU.lock
while true; do
  while busy; do sleep 30; done
  flock -w 86400 9 || { echo "no lock in 24 h"; exit 3; }
  if busy; then flock -u 9; sleep 30; continue; fi
  break
done
echo "$(date -u +%T) GPU.lock taken"
python3 -I $H/live/live3.py nr
echo "$(date -u +%T) nr rc=$?"
