#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=models/uncertainty/solver.prototxt
