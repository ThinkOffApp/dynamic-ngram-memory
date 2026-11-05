#!/bin/bash
# gpucheck driver for one arm on asus3: prep, start, a due note, replies and hashes before, the refused night, after.
set -u
ARM=$1
G=$HOME/tf-promote/gpucheck; cd $G || exit 1
P="python3 -I $G/gpucheck.py $ARM"
trap '[ -f $G/$ARM/server.pid ] && $P stop' EXIT
[ "${2:-}" = skip-prep ] || $P prep || exit 1
$P start || exit 1
$P promotion start || exit 1
$P note || exit 1
$P retrieve || exit 1
$P ask before || exit 1
$P hashes before || exit 1
$P promotion before || exit 1
$P night || exit 1
$P hashes after || exit 1
$P promotion after || exit 1
$P ask after || exit 1
$P stop || exit 1
$P hashes after-stop
echo "RUNDONE $ARM"
