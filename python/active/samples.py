import config

import numpy as np
import caffe
import lmdb

import utils
import db


# class SamplePool(object):
#
#     def __init__(self):
#         pass
#
#
# class Policy(object):
#     pass
#
# class Scorer(object):
#     pass
#
#
# def setup():
#
#     db_path = ''
#     scorer = Scorer()
#     policy = Policy()
#
#     pool = SamplePool(db_path, scorer, policy)
#
#     epochs = 10
#
#     for e in xrange(epochs):
#         # run epoch
#         for batch in pool:
#             pass
#
#         SamplePool.reset()


class Dataset(object):

    def __init__(self, db_path, batch_size, skip_keys=None):
        self.map_size = 2**40
        self.env = None
        self.db_path = None
        self.batch_size = None
        self.init(db_path, batch_size)

        if skip_keys is None:
            self.skip_keys = set()
        else:
            self.skip_keys = skip_keys

    def init(self, db_path=None, batch_size=None):
        if self.is_opened():
            self.close()

        if db_path is not None:
            self.db_path = db_path
            self.size = db.size(self.db_path)
            self.entry_shape = db.entry_shape(self.db_path)
            self.shape = tuple([self.size] + list(self.entry_shape))

        if batch_size is not None:
            self.batch_size = batch_size

    def __len__(self):
        return self.size - len(self.skip_keys)

    def num_batches(self):
        return len(self) / self.batch_size

    def open(self):
        if self.is_opened():
            self.close()

        self.env = lmdb.open(self.db_path, readonly=True, map_size=self.map_size)

    def close(self):
        self.env.close()
        self.env = None

    def is_opened(self):
        return self.env is not None

    def __iter__(self):
        X = np.zeros([self.batch_size] + list(self.entry_shape), dtype=np.float32)
        y = np.zeros(self.batch_size, dtype=np.int32)
        keys = np.zeros(self.batch_size, dtype=np.object)

        with self.env.begin() as txn:
            index = 0
            for key, raw_datum in txn.cursor():
                if self.skip_keys is not None:
                    if key in self.skip_keys:
                        continue

                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                X[index] = flat_x.reshape(datum.channels, datum.height, datum.width)
                y[index] = datum.label
                keys[index] = key
                index += 1

                if index == self.batch_size:
                    yield X, y, keys
                    index = 0


def criterium(uncertainty, correct, keys, labels, num_samples_to_choose):

    # keys = keys[correct]
    # uncertainty = uncertainty[correct]
    sorted_keys = sorted(zip(uncertainty, keys), key=lambda x: x[0], reverse=True)

    for i in xrange(0, len(sorted_keys), len(sorted_keys)/10):
        print sorted_keys[i]

    sorted_keys = [x[1] for x in sorted_keys if x[0] > config.UNCERTAINTY_THRESOLD]
    print 'Chosen {0} keys with threshold {1}'.format(len(sorted_keys), config.UNCERTAINTY_THRESOLD)
    return sorted_keys[:num_samples_to_choose]


def criterium_balanced(uncertainty, correct, keys, labels, num_samples_to_choose):

    classes = {}
    for label, key, unc in zip(labels, keys, uncertainty):
        if label not in classes:
            classes[label] = []

        if unc > config.UNCERTAINTY_THRESOLD:
            classes[label].append((unc, key))

    for key in classes.keys():
        classes[key].sort(key=lambda x: x[0], reverse=True)
        classes[key] = [x[1] for x in classes[key]]

    samples_per_class = 2 * num_samples_to_choose / len(classes.keys())
    for key in classes.keys():
        classes[key] = classes[key][:samples_per_class]

    chosen_keys = list(utils.roundrobin(*classes.values()))

    print 'Chosen {0} keys with threshold {1}'.format(len(chosen_keys), config.UNCERTAINTY_THRESOLD)
    return chosen_keys[:num_samples_to_choose]


def choose_active(net, dataset, num_batches_to_choose):
    print 'Choosing active samples....'

    total_num_batches = dataset.num_batches()
    total_num_samples = len(dataset)
    num_batches_to_choose = min(num_batches_to_choose, total_num_batches)

    # reserve storage for outputs
    uncert = np.zeros(total_num_samples, dtype=np.float32)
    correct = np.zeros(total_num_samples, dtype=bool)
    keys = np.zeros(total_num_samples, dtype=np.object)
    labels = np.zeros(total_num_samples, dtype=np.uint8)

    dataset.open()
    for batch_num, batch in enumerate(dataset):
        utils.wait_bar('Batch number', '', batch_num + 1, total_num_batches)
        beg = batch_num * net.batch_size
        end = beg + net.batch_size

        X, y, batch_keys = batch
        output = net.forward(X)[0]
        uncert[beg:end] = utils.entropy(output)
        correct[beg:end] = np.equal(output.argmax(axis=1), y)
        keys[beg:end] = batch_keys
        labels[beg:end] = y

    dataset.close()
    print
    print 'Accuracy = {0}, mean uncertainty = {1}'.format(correct.mean(), uncert.mean())

    num_samples_to_choose = num_batches_to_choose * net.batch_size
    chosen_keys = criterium_balanced(uncert, correct, keys, labels, num_samples_to_choose)

    num_to_return = (len(chosen_keys) / net.batch_size) * net.batch_size
    chosen_keys = chosen_keys[:num_to_return]
    print 'Returning {0} new samples'.format(len(chosen_keys))
    return chosen_keys




