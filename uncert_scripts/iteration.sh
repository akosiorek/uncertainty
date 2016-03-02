#!/usr/bin/env bash
#
# Performs a single iteration: training, testing, plotting and renaming results
#
#
./models/uncertainty/train.sh
./uncert_scripts/test.sh 100 5000
mv results/ $1
