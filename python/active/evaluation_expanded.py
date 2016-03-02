#!/usr/bin/env python

import numpy as np
import os
import sys

os.environ['GLOG_minloglevel'] = '1'
import caffe

import utils
from evaluation import NetEvaluation

UNCERTS = ['max_unc', 'entropy_conf', 'entropy_ip2', '2_max_ip2', 'entropy_weighted', '2_max_weighted']
NUM_UNC = len(UNCERTS)


class EvaluationExpanded(NetEvaluation):

    def __init__(self, net_path, db_path, snapshot_path, results_folder, every_iter=None):
        super(EvaluationExpanded, self).__init__(net_path, db_path, snapshot_path, results_folder, every_iter)
        self.output_folders = [os.path.join(results_folder, output_name) for output_name in UNCERTS]

    def prepare_output_folder(self):
        super(EvaluationExpanded, self).prepare_output_folder()
        for output_folder in self.output_folders:
            os.mkdir(output_folder)

    def init_output_storage(self):
        self.uncertainty = np.zeros((self.db_size, NUM_UNC), dtype=np.float32)
        self.correct = np.zeros(self.db_size, dtype=np.int8)

    def process_intermediate_output(self, itr, X, y, net):
        beg = itr * self.batch_size
        end = beg + self.batch_size

        weighted = utils.softmax(net.blobs["weighted_input"].data)
        ip2 = utils.softmax(net.blobs["ip2"].data)
        confidence = net.blobs["confidence"].data

        y_predicted = weighted.argmax(axis=1)

        self.uncertainty[beg:end, 0] = 1-confidence[xrange(y_predicted.shape[0]), y_predicted]
        self.uncertainty[beg:end, 1] = utils.entropy(confidence)
        self.uncertainty[beg:end, 2] = utils.entropy(ip2)
        self.uncertainty[beg:end, 3] = utils.second_max(ip2)
        self.uncertainty[beg:end, 4] = utils.entropy(weighted)
        self.uncertainty[beg:end, 5] = utils.second_max(weighted)

        self.correct[beg:end] = np.equal(y, y_predicted)

    def process_output(self, snapshot_num):
        for output_num, outut_folder in enumerate(self.output_folders):
            uncert_path = os.path.join(outut_folder, 'uncert_{0}.txt'.format(snapshot_num))
            label_path = os.path.join(outut_folder, 'label_{0}.txt'.format(snapshot_num))

            utils.write_to_file(uncert_path, self.uncertainty[:, output_num])
            utils.write_to_file(label_path, self.correct)

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
    evaluation = EvaluationExpanded(net_path, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()