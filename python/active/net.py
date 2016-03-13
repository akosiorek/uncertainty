import os
import numpy as np

os.environ['GLOG_minloglevel'] = '1'
import caffe

import db
import utils
import proto
import config


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
        print 'Reading mean file from {}'.format(mean_path)
        self.mean_path = mean_path
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(self.mean_path, 'rb').read())
        arr = np.array(caffe.io.blobproto_to_array(blob))
        self.mean = arr[0]

    def prepare_deploy_net(self, net_path):
        deploy_net = utils.create_temp_path(net_path + '.deploy' + config.POSTFIX)
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