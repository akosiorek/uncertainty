#!/usr/bin/env python

import os
import sys
import numpy as np

os.environ['GLOG_minloglevel'] = '1'
import caffe
import lmdb

import db
import utils
import proto
import samples
import learn


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

        self.batch_size, mean_path = proto.get_batch_mean_from_net(self.net_path)
        self.input_shape = db.entry_shape(self.db_path)
        self.db_size = db.size(self.db_path)
        self.num_batches = self.db_size / self.batch_size
        self.deploy_net_path = os.path.join('/tmp', os.path.basename(self.net_path) + '.deploy' + learn.POSTFIX)

        self.mean = samples.read_meanfile(mean_path)
        proto.prepare_deploy_net(self.net_path, self.deploy_net_path, self.batch_size, self.input_shape)

        if os.path.isdir(self.snapshot_path):
            self.snapshots = utils.get_snapshot_files(self.snapshot_path, every_iter=self.every_iter)
        else:
            self.snapshots = ((utils.get_snapshot_number(self.snapshot_path), self.snapshot_path),)

        self.prepare_output_folder()
        self.init_output_storage()
        self.initialized = True

    def evaluate(self):
        self.init()
        print 'Evaluating...'

        for snapshot_num, snapshot_path in self.snapshots:
            print 'Processing snapshot #{0}'.format(snapshot_num)
            # output = evaluate(self.deploy_net_path, snapshot_path, self.mean, self.db_path, self.batch_size)
            self.compute(snapshot_path)
            self.process_output(snapshot_num)

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

                net.forward(data=X)
                self.process_intermediate_output(itr, X, y, net)
        print

    def prepare_output_folder(self):
        utils.clear_dir(self.results_folder)

    def init_output_storage(self):
        self.uncertainty = np.zeros(self.db_size, dtype=np.float32)
        self.correct = np.zeros(self.db_size, dtype=np.int8)

    def process_intermediate_output(self, itr, X, y, net):
        beg = itr * self.batch_size
        end = beg + self.batch_size

        y_predicted = net.blobs["ip2"].data.argmax(axis=1)
        self.uncertainty[beg:end] = utils.entropy(utils.softmax(net.blobs["ip2"].data))
        self.correct[beg:end] = np.equal(y, y_predicted)

    def process_output(self, snapshot_num):
        uncert_path = os.path.join(self.results_folder, 'uncert_{0}.txt'.format(snapshot_num))
        label_path = os.path.join(self.results_folder, 'label_{0}.txt'.format(snapshot_num))

        utils.write_to_file(uncert_path, self.uncertainty)
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
    evaluation = NetEvaluation(net_path, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()

