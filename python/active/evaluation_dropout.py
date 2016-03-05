#!/usr/bin/env python

import numpy as np
import os
import sys

os.environ['GLOG_minloglevel'] = '1'
import caffe
import lmdb

import samples
import utils
from evaluation import NetEvaluation


DROPOUT_ITERS = 10
VARIANTS = ['vote', 'weighted']
NUM = len(VARIANTS)


class EvaluationDropout(NetEvaluation):

    def __init__(self, net_path, db_path, snapshot_path, results_folder, every_iter=None):
        super(EvaluationDropout, self).__init__(net_path, db_path, snapshot_path, results_folder, every_iter)
        self.output_folders = [os.path.join(results_folder, output_name) for output_name in VARIANTS]

    def prepare_output_folder(self):
        super(EvaluationDropout, self).prepare_output_folder()
        for output_folder in self.output_folders:
            os.mkdir(output_folder)

    def init_output_storage(self):
        self.uncertainty = np.zeros((NUM, self.db_size), dtype=np.float32)
        self.correct = np.zeros((NUM, self.db_size), dtype=np.int8)

    def compute(self, snapshot_path):
        net = caffe.Net(self.deploy_net_path, snapshot_path, caffe.TEST)
        env = lmdb.open(self.db_path, readonly=True)

        with env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()

            for itr in xrange(self.num_batches):
                utils.wait_bar('Evaluating batch ', '...', itr+1, self.num_batches)

                X, y, keys = samples.build_batch(cursor, self.batch_size, self.input_shape)
                X -= self.mean

                output = np.zeros([2] + list(net.blobs["ip2"].data.shape), dtype=np.float32)
                for i in xrange(DROPOUT_ITERS):
                    net.forward(data=X)

                    output[0, :, :] += utils.softmax(net.blobs["ip2"].data)

                    argmax = net.blobs["ip2"].data.argmax(axis=1)
                    lin_index = xrange(argmax.shape[0])
                    output[1, lin_index, argmax] +=1

                output /= DROPOUT_ITERS

                beg = itr * self.batch_size
                end = beg + self.batch_size

                for j in xrange(NUM):
                    probs = output[j, :, :]
                    self.uncertainty[j, beg:end] = utils.entropy(probs)
                    self.correct[j, beg:end] = np.equal(y, probs.argmax(axis=1))

                    # print j, self.correct[j, beg], self.uncertainty[j, beg]

    def process_output(self, snapshot_num):
        for output_num, outut_folder in enumerate(self.output_folders):
            uncert_path = os.path.join(outut_folder, 'uncert_{0}.txt'.format(snapshot_num))
            label_path = os.path.join(outut_folder, 'label_{0}.txt'.format(snapshot_num))

            utils.write_to_file(uncert_path, self.uncertainty[output_num, :])
            utils.write_to_file(label_path, self.correct[output_num, :])


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
    evaluation = EvaluationDropout(net_path, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()