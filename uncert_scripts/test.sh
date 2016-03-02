#!/usr/bin/env bash
# Perform tests of multiple network snapshots given by $(seq $1 100 $2)
# It creates a folder structure, saves correctness and uncertainties in
# text files and visualizes them.



model=-model=models/uncertainty/net.prototxt
weights=-weights=models/uncertainty/snapshot/snapshot_iter_
scripts=uncert_scripts
python_scripts=python/uncertainty

./$scripts/make_folders.sh

for n in $(seq $1 100 $2)
do
	real_weights=$weights$n.caffemodel.h5
	./build/tools/caffe test $model $real_weights --iterations=5 -gpu=0
	./$python_scripts/plot_uncert.py uncert.txt label.txt
	mv uncert.txt results/numbers/uncert_$n.txt
	mv label.txt results/numbers/label_$n.txt

	mv uncert_positive.png results/plots/positive/positive_$n.png  
	mv uncert_negative.png results/plots/negative/negative_$n.png
done


./$python_scripts/visualize.py results/numbers
mv sample_uncert_evolution.png results/plots
mv uncert_and_acc_plot.png results/plots 

cp models/uncertainty/*.prototxt results/model/
