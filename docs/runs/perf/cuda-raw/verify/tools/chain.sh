#!/bin/bash
# The verify chain after lanesq.sh: CLI rounds, CLI bursts, 4,096-token prompts, two server series, nsys, window
# profiles; one stage at a time, each taking GPU.lock itself. A file raw/v/STOP stops it before the next run.
set -u
source ~/tf-pocket-cuda/tools/v/common.sh
V=$H/tools/v; R=$H/raw/v
while pgrep -f "tools/v/lanesq.sh" >/dev/null; do sleep 20; done
echo "$(date -u +%T) lanesq finished; df: $(df -h / | tail -1)"
stage() { stop_check; local out=$1; shift; echo "$(date -u +%T) stage $* -> $out"; "$@" > $out 2>&1; echo "$(date -u +%T) stage end rc=$?"; }
stage $R/rounds-r1.out bash $V/rounds2.sh r1 4 60
stage $R/cli-c1.out bash $V/clib.sh c1 6 90
stage $R/prefill-p1.out bash $V/prefill2.sh p1 3 60
stage $R/servers-s1.out bash $V/servers2.sh s1 120 "bT aT aA bA bE aE" openai,openai256,off,conc,conc14,facts
stage $R/servers-s2.out bash $V/servers2.sh s2 120 "aE bE bA aA aT bT" openai,openai256,off,conc,conc14
stage $R/nsys-n1.out bash $V/nsys2.sh n1 "after:T:nd after:T:d after:E:nd before:T:nd after:A:nd"
stage $R/wp-w1.out bash $V/wp2.sh w1 30 "before:T after:T after:E before:E after:A before:A"
echo "$(date -u +%T) chain done; df: $(df -h / | tail -1)"
