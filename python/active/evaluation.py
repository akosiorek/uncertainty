#!/usr/bin/env python

import os
import sys
import shutil
import numpy as np

os.environ['GLOG_minloglevel'] = '1'
import caffe

import lmdb
import utils
import proto
import samples
import active_learning


class NetEvaluation(object):

    def __init__(self, net_path, db_path, snapshot_path, results_folder, every_iter=None):
        self.net_path = net_path
        self.db_path = db_path
        self.snapshot_path = snapshot_path
        self.results_folder = results_folder
        self.every_iter = every_iter
        self.initialized = False

    def init(self):
        if self.initialized:
            return

        print 'Initializing...'

        self.batch_size, mean_path = proto.get_batch_mean_from_net(net_path)
        self.input_shape = samples.entry_shape(db_path)
        self.db_size = samples.len_db(self.db_path)
        self.num_batches = self.db_size / self.batch_size
        self.deploy_net_path = os.path.join('/tmp', os.path.basename(self.net_path) + '.deploy' + active_learning.POSTFIX)

        self.mean = samples.read_meanfile(mean_path)
        proto.prepare_deploy_net(self.net_path, self.deploy_net_path, self.batch_size, self.input_shape)

        if os.path.isdir(self.snapshot_path):
            self.snapshots = utils.get_snapshot_files(self.snapshot_path, every_iter=self.every_iter)
        else:
            self.snapshots = ((utils.get_snapshot_number(self.snapshot_path), self.snapshot_path),)

        if os.path.exists(self.results_folder):
            shutil.rmtree(self.results_folder)
        os.mkdir(self.results_folder)

        self.initialized = True

    def evaluate(self):
        self.init()
        print 'Evaluating...'

        for snapshot_num, snapshot_path in self.snapshots:
            print 'Processing snapshot #{0}'.format(snapshot_num)
            # output = evaluate(self.deploy_net_path, snapshot_path, self.mean, self.db_path, self.batch_size)
            output = self.compute(snapshot_path)
            self.process_output(snapshot_num, output)

    def process_output(self, snapshot_num, output):
        uncert, correct = output
        uncert_path = os.path.join(results_folder, 'uncert_{0}.txt'.format(snapshot_num))
        label_path = os.path.join(results_folder, 'label_{0}.txt'.format(snapshot_num))

        utils.write_to_file(uncert_path, uncert)
        utils.write_to_file(label_path, correct)

    def compute(self, snapshot_path):
        uncertainty = np.zeros(self.db_size, dtype=np.float32)
        correct = np.zeros(self.db_size, dtype=np.int8)

        net = caffe.Net(self.deploy_net_path, snapshot_path, caffe.TEST)
        env = lmdb.open(self.db_path, readonly=True)

        with env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            beg, end = 0, self.batch_size

            for index in xrange(self.num_batches):
                utils.wait_bar('Evaluating batch ', '...', index+1, self.num_batches)

                X, y, keys = samples.build_batch(cursor, self.batch_size, self.input_shape)
                X -= self.mean

                net.forward(data=X)
                y_predicted = net.blobs["ip2"].data.argmax(axis=1)
                uncertainty[beg:end] = utils.entropy(utils.softmax(net.blobs["ip2"].data))
                correct[beg:end] = np.equal(y, y_predicted)
                beg += self.batch_size
                end += self.batch_size

        print
        return uncertainty, correct


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
    evaluation = NetEvaluation(net_path, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()

