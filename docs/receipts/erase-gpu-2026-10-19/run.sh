#!/bin/bash
# E-erase driver on gb10-a: prep, A (before), erase, B (after), restart, C (after restart). Stops its server on any exit.
set -u
D=$HOME/tf-promote/eerase-20261019; cd $D || exit 1
P="python3 -I $D/eerase.py"
trap '[ -f $D/server.pid ] && $P stop' EXIT
$P prep || exit 1
$P start || exit 1
$P promotion A && $P ask A || exit 1
$P erase || exit 1
sleep 5
$P promotion B && $P ask B || exit 1
$P stop || exit 1
$P hashes after-erase
$P start || exit 1
$P promotion C && $P ask C || exit 1
$P stop
$P hashes C
echo EERASEDONE
