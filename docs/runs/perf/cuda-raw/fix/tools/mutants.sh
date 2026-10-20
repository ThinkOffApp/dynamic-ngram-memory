#!/bin/bash
# The pocket kernel test against deliberately broken kernels: each mutant must FAIL. Builds in ~/tf-pocket-cuda/mut.
set -u
H=$HOME/tf-pocket-cuda
rsync -a --delete --exclude zig-out --exclude .zig-cache $H/src/ $H/mut/
cd $H/mut
export ZIG_GLOBAL_CACHE_DIR=$H/.zig-global
F=zig/kernels/cuda/nemotron_ops.cu
cp $F /tmp/nemotron_ops.cu.orig.$$
run() {
  name=$1
  echo "== $name: $(diff /tmp/nemotron_ops.cu.orig.$$ $F | grep "^>" | head -2)"
  nice -n 10 ~/pr144-review/zig/zig build install -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8 -j8 > $H/raw/fix/mut-$name.build 2>&1 || { echo "build failed"; return; }
  $H/tools/gpu.sh $H/raw/fix/mut-$name.log zig-out/bin/tf-cuda-test pocket 0 >/dev/null
  grep -E "^(PASS|FAIL|DIFFER)" $H/raw/fix/mut-$name.log | head -4
  cp /tmp/nemotron_ops.cu.orig.$$ $F
}
python3 - $F <<PY
import sys; p=sys.argv[1]; s=open(p).read()
a="                for (int b = 0; b < B; ++b)\n                    if (j0 + b < n) s[t] += a[t * pocket_width + j0 + b] * w[b];"
assert a in s; open(p,"w").write(s.replace(a,"                for (int b = B - 1; b >= 0; --b)\n                    if (j0 + b < n) s[t] += a[t * pocket_width + j0 + b] * w[b];"))
PY
run out_batch_reversed
python3 - $F <<PY
import sys; p=sys.argv[1]; s=open(p).read()
a="                const float z = fmaxf(v, 0.f);\n                act[static_cast<size_t>(r0 + t) * pocket_width + j0 + q] = z * z;"
assert a in s; open(p,"w").write(s.replace(a,"                const float z = v;\n                act[static_cast<size_t>(r0 + t) * pocket_width + j0 + q] = z * z;"))
PY
run in_no_relu
python3 - $F <<PY
import sys; p=sys.argv[1]; s=open(p).read()
a="for (int t = 0; t < R; ++t) s[q][t] += u[q] * x[t];"
assert a in s; open(p,"w").write(s.replace(a,"for (int t = 0; t < R; ++t) s[q][t] = __fadd_rn(s[q][t], __fmul_rn(u[q], x[t]));"))
PY
run in_no_fma
python3 - $F <<PY
import sys; p=sys.argv[1]; s=open(p).read()
a="    const int k = blockIdx.x * blockDim.x + threadIdx.x;\n    if (k >= d) return;\n    float s[R];"
assert a in s; open(p,"w").write(s.replace(a,"    const int k = blockIdx.x * blockDim.x + threadIdx.x;\n    if (k >= d - 1) return;\n    float s[R];"))
PY
run out_last_dropped
rm -f /tmp/nemotron_ops.cu.orig.$$
echo done
