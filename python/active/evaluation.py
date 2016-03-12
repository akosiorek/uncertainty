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


class Net(caffe.Net):
    def __init__(self, net_path, trained_path=None, output_layers=[], is_deploy=False, mean_path=None,
                 batch_size=None, input_shape=None):

        self.net_path = net_path
        self.trained_path = trained_path
        self.is_deploy = is_deploy
        self.net = None
        self.mean = None
        self.output_layers = output_layers
        self.batch_size = batch_size
        self.mean_path = mean_path
        self.input_shape = input_shape

        if self.batch_size is None or self.mean_path is None:
            batch_size, mean_path = proto.get_batch_mean_from_net(self.net_path)

            if self.batch_size is None:
                self.batch_size = batch_size

            if self.mean_path is None:
                self.mean_path = mean_path

    def load_model(self, model_path):
        self.trained_path = model_path
        self.net = None

    def load_mean(self, mean_path):
        self.mean_path = mean_path
        self.mean = samples.read_meanfile(self.mean_path)

    def prepare_deploy_net(self, net_path):
        deploy_net = utils.create_temp_path(net_path + '.deploy' + learn.POSTFIX)
        proto.prepare_deploy_net(net_path, deploy_net, self.batch_size, self.input_shape)
        return deploy_net

    def prepare_deploy(self):

        if self.input_shape is None:
            db_path = proto.get_db_from_net(self.net_path)
            self.input_shape = db.entry_shape(db_path)

        self.net_path = self.prepare_deploy_net(self.net_path)
        self.is_deploy = True

    def init(self):
        if not self.is_deploy:
            self.prepare_deploy()

        if self.mean is None and self.mean_path is not None:
            self.load_mean(self.mean_path)

    def forward(self, X, *args, **kwargs):
        self.init()
        if self.net is None:
            if self.trained_path is None:
                raise ValueError('No trained model has been given.')

            self.net = caffe.Net(self.net_path, self.trained_path, caffe.TEST)

        if self.mean is not None:
            X -= self.mean

        return self.forward_impl(X, *args, **kwargs)

    def forward_impl(self, X, *args, **kwargs):
        self.net.forward(data=X, *args, **kwargs)
        return [self.net.blobs[name].data for name in self.output_layers]


class DropoutNet(Net):

    # TODO: possible bug: uncertainty seems really high
    @staticmethod
    def aggregate_mean(X):
        for i in xrange(X.shape[0]):
            X[i] = utils.softmax(X[i])
        return X.mean(axis=0)

    @staticmethod
    def aggregate_vote(X):
        lin_index = xrange(X.shape[1])
        output = np.zeros(X.shape[1:], dtype=np.float32)
        for i in xrange(X.shape[0]):
            argmax = X[i].argmax(axis=1)
            output[lin_index, argmax] += 1
        return output

    def __init__(self, net_path, dropout_iters, aggregate='vote', *args, **kwargs):
        super(DropoutNet, self).__init__(net_path, *args, **kwargs)
        self.droput_iters = dropout_iters

        assert len(self.output_layers) == 1, 'DropoutNet supports only a single output layer'

        if aggregate == 'mean':
            self.aggregate = DropoutNet.aggregate_mean
        elif aggregate == 'vote':
            self.aggregate = DropoutNet.aggregate_vote
        else:
            raise ValueError('Unknown aggregation method: {}'.format(aggregate))

    def forward_impl(self, X, *args, **kwargs):
        r = super(DropoutNet, self).forward_impl(X, *args, **kwargs)

        result = []
        for i in xrange(len(r)):
            shape = [self.droput_iters] + list(r[i].shape)
            mat = np.ndarray(shape, dtype=np.float32)
            mat[0, :, :] = r[i]
            result.append(mat)

        for i in xrange(1, self.droput_iters - 1):
            r = super(DropoutNet, self).forward_impl(X, *args, **kwargs)
            for j in xrange(len(result)):
                result[j][i, :, :] = r[j]

        for j in xrange(len(result)):
                result[j] = self.aggregate(result[j])

        return result


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
    net = Net(net_path, output_layers=['ip2'])
    evaluation = NetEvaluation(net, db_path, snapshot_folder, results_folder, every_iter)
    evaluation.evaluate()

