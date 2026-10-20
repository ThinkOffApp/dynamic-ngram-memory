#!/bin/bash
# One GPU job under GPU.lock, refused while another engine, test, nsys or the dedup job runs: gpu.sh LOG CMD...
set -u
exec 9>~/tf-pocket-cuda/GPU.lock
flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }
LOG=$1; shift
if pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null || pgrep -f "^python3 [^ ]*dedup_identical_copies" >/dev/null; then
  echo "$(date -u +%T) refusing: another GPU or dedup process"; pgrep -af "tensorfold|tf-cuda-test|nsys|dedup_identical"; exit 4
fi
T0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
echo "START $(date -u +%FT%TZ) temp=$T0 cmd: $*" > $LOG
"$@" >> $LOG 2>&1
rc=$?
echo "END $(date -u +%FT%TZ) rc=$rc" >> $LOG
echo "rc=$rc"
