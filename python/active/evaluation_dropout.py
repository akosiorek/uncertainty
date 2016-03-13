#!/usr/bin/env python

import os
import sys

os.environ['GLOG_minloglevel'] = '1'
import caffe

import net
import evaluation
import config

if __name__ == '__main__':
    args = sys.argv[1:]

    net_path = args[0]
    db_path = args[1]
    snapshot_folder = args[2]
    results_folder = args[3]

    if len(args) > 4:
        every_iter = int(args[4])
    else:
        every_iter = None

    caffe.set_mode_gpu()
    net = net.DropoutNet(net_path, config.DROPOUT_ITERS, aggregate='mean', output_layers=config.OUTPUT_LAYERS)
    evaluation = evaluation.NetEvaluation(net, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()