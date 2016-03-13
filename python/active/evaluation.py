#!/usr/bin/env python

import os
import sys
import numpy as np

os.environ['GLOG_minloglevel'] = '1'
import caffe
import lmdb

import db
import utils
import samples


class NetEvaluation(object):

    def __init__(self, net, db_path, snapshot_path, results_folder, every_iter=None):
        self.net = net
        self.db_path = db_path
        self.snapshot_path = snapshot_path
        self.results_folder = results_folder
        self.every_iter = every_iter
        self.initialized = False

    def init(self):
        print 'Initializing...'

        self.db_size = db.size(self.db_path)
        self.num_batches = self.db_size / self.net.batch_size
        self.net.input_shape = db.entry_shape(self.db_path)

        if os.path.isdir(self.snapshot_path):
            self.snapshots = utils.get_snapshot_files(self.snapshot_path, every_iter=self.every_iter)
        else:
            self.snapshots = ((utils.get_snapshot_number(self.snapshot_path), self.snapshot_path),)

        self.prepare_output_folder()
        self.init_output_storage()

        self.net.init()
        self.initialized = True

    def evaluate(self):
        if not self.initialized:
            self.init()

        print 'Evaluating...'

        for snapshot_num, snapshot_path in self.snapshots:
            print 'Processing snapshot #{0}'.format(snapshot_num)
            self.compute(snapshot_path)
            self.process_output(snapshot_num)

    def compute(self, snapshot_path):
        self.net.load_model(snapshot_path)
        env = lmdb.open(self.db_path, readonly=True)

        with env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()

            for itr in xrange(self.num_batches):
                utils.wait_bar('Evaluating batch ', '...', itr+1, self.num_batches)

                X, y, keys = samples.build_batch(cursor, self.net.batch_size, self.net.input_shape)
                output = self.net.forward(X)
                self.process_intermediate_output(itr, X, y, output)
        print

    def prepare_output_folder(self):
        utils.clear_dir(self.results_folder)

    def init_output_storage(self):
        self.uncertainty = np.zeros(self.db_size, dtype=np.float32)
        self.correct = np.zeros(self.db_size, dtype=np.int8)

    def process_intermediate_output(self, itr, X, y, output):
        beg = itr * self.net.batch_size
        end = beg + self.net.batch_size

        y_predicted = output[0].argmax(axis=1)
        self.uncertainty[beg:end] = utils.entropy(utils.softmax(output[0]))
        self.correct[beg:end] = np.equal(y, y_predicted)

    def process_output(self, snapshot_num):
        uncert_path = os.path.join(self.results_folder, 'uncert_{0}.txt'.format(snapshot_num))
        label_path = os.path.join(self.results_folder, 'label_{0}.txt'.format(snapshot_num))

        utils.write_to_file(uncert_path, self.uncertainty)
        utils.write_to_file(label_path, self.correct)


if __name__ == '__main__':
    import net

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
    net = net.Net(net_path, output_layers=['ip2'])
    evaluation = NetEvaluation(net, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()

