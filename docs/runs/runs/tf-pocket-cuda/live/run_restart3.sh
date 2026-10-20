#!/bin/bash
# Phase 3 folders (after lesson B2, layer 22 back to 0) served again: before on its own, tip on its own, e987544e (src) on the mutant run's.
set -u
H=~/tf-pocket-cuda
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null; }
exec 9>$H/GPU.lock
while true; do
  while busy; do sleep 30; done
  flock -w 86400 9 || { echo "no lock in 24 h"; exit 3; }
  if busy; then flock -u 9; sleep 30; continue; fi
  break
done
echo "$(date -u +%T) GPU.lock taken"
python3 -I $H/live/live3.py restart before before
python3 -I $H/live/live3.py restart tip tip
python3 -I $H/live/live3.py restart inst tip
python3 -I $H/live/live3.py restart mutant src
echo "$(date -u +%T) done"
