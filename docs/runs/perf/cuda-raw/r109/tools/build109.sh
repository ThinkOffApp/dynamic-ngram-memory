#!/bin/bash
# 1.0.9 receipt: build r109/before (c36b536d) and r109/after (76fe1de4) like review/tools/build.sh (install, then native),
# then zig build test on each, then binary and fatbin sha256s. CPU only, nice 19, -j4. usage: build109.sh
set -u
H=$HOME/tf-pocket-cuda/r109
O=$H/raw
export ZIG_GLOBAL_CACHE_DIR=$HOME/tf-pocket-cuda/.zig-global
Z=~/pr144-review/zig/zig
FL="-Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.9"
echo "zig $($Z version) nvcc $(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)"
free -g | head -2
for T in before after; do
  cd $H/$T
  for step in install native; do
    echo "START $T $step $(date -u +%FT%TZ)"
    nice -n 19 $Z build $step $FL -j4 > $O/build-$T-$step.log 2>&1
    rc=$?
    echo "END $T $step $(date -u +%FT%TZ) rc=$rc"
    [ $rc -eq 0 ] || { tail -30 $O/build-$T-$step.log; exit $rc; }
  done
done
for T in before after; do
  cd $H/$T
  echo "START $T test $(date -u +%FT%TZ)"
  nice -n 19 $Z build test $FL -j4 --summary all > $O/test-$T.log 2>&1
  echo "END $T test $(date -u +%FT%TZ) rc=$? $(grep -E "Build Summary" $O/test-$T.log | tail -1)"
done
for T in before after; do
  (cd $H/$T && find zig-out -type f \( -path "*/bin/*" -o -name "*.fatbin" \) -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $O/bin-$T.sha256
done
echo "$(date -u +%T) done"
