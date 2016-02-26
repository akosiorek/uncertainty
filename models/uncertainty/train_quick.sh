#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=models/uncertainty/solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=models/uncertainty/solver_lr1.prototxt \
  --snapshot=models/uncertainty/snapshot/snapshot_iter_4000.solverstate.h5
