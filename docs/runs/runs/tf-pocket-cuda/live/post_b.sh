#!/bin/bash
# After the live run: the -0.0 test commit's files into src, src rebuilt as before (install, native), then
# tf-cuda-test pocket and zig build test; then a mutant without the launcher's skip in srcB, which must fail.
set -u
H=~/tf-pocket-cuda
L=$H/live
Z=~/pr144-review/zig/zig
export ZIG_GLOBAL_CACHE_DIR=$H/.zig-global
FL="-Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8"
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null; }
while pgrep -f "run_live.sh" >/dev/null; do sleep 15; done
grep -q "after rc=0" $L/run_live.log || { echo "$(date -u +%T) the live run did not end with after rc=0: stopping"; exit 5; }
exec 9>$H/GPU.lock
while true; do
  while busy; do echo "$(date -u +%T) waiting: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test)"; sleep 30; done
  flock -w 86400 9 || { echo "$(date -u +%T) no lock in 24 h"; exit 3; }
  if busy; then flock -u 9; echo "$(date -u +%T) lock taken but another GPU process runs: released"; sleep 30; continue; fi
  break
done
echo "$(date -u +%T) GPU.lock taken"
F="zig/src/families/nemotron/cuda_forward.zig zig/src/families/nemotron/cuda_kernels.zig zig/tests/cuda/pocket_tests.zig"
for f in $F; do cp $H/srcB/$f $H/src/$f; done
(cd $H/src && sha256sum $F)
cd $H/src
for step in install native; do
  echo "START $step $(date -u +%FT%TZ)"
  nice -n 10 $Z build $step $FL -j8 > $L/build-src-$step.log 2>&1
  rc=$?; echo "END $step $(date -u +%FT%TZ) rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
sha256sum zig-out/bin/tensorfold zig-out/bin/tf-cuda-test zig-out/native/bin/tensorfold-native
echo "START pocket $(date -u +%FT%TZ) temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)"
zig-out/bin/tf-cuda-test pocket > $L/pocket-test.log 2>&1
echo "END pocket rc=$?"; grep -E "^(PASS|FAIL|SKIP|DIFFER)" $L/pocket-test.log
echo "START zig build test $(date -u +%FT%TZ)"
nice -n 10 $Z build test $FL -j8 --summary all > $L/zig-build-test.log 2>&1
echo "END zig build test rc=$?"; tail -5 $L/zig-build-test.log
# the control: without the launcher's skip the -0.0 case must fail
cd $H/srcB
sed -i 's|^        if (p.used == 0) return; // no neuron in use: nothing launched, so the output keeps its bits, an exact -0.0 too$|        // MUTANT: the skip removed|' zig/src/families/nemotron/cuda_kernels.zig
grep -n "MUTANT" zig/src/families/nemotron/cuda_kernels.zig || { echo "mutant not applied"; exit 4; }
nice -n 10 $Z build install $FL -j8 > $L/build-mutant.log 2>&1; echo "mutant build rc=$?"
zig-out/bin/tf-cuda-test pocket > $L/pocket-test-mutant.log 2>&1
echo "mutant pocket rc=$?"; grep -E "^(PASS|FAIL|SKIP)" $L/pocket-test-mutant.log
cp $H/src/zig/src/families/nemotron/cuda_kernels.zig zig/src/families/nemotron/cuda_kernels.zig
cmp zig/src/families/nemotron/cuda_kernels.zig $H/src/zig/src/families/nemotron/cuda_kernels.zig && echo "srcB restored"
echo "$(date -u +%T) done"
