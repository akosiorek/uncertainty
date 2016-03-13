import numpy as np
import caffe
import lmdb

import utils
import db


def build_batch(cursor, batch_size, input_shape, skip_keys=None):

    X = np.zeros([batch_size] + list(input_shape), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.int32)
    keys = np.zeros(batch_size, dtype=np.object)

    first_key = None
    index = 0
    while index < batch_size:
        key, raw_datum = cursor.item()
        db.move_cursor_circular(cursor)

        # check if we've seen this key already
        if first_key is None:
            first_key = key
        elif first_key == key:
            return None

        if skip_keys is not None:
            if key in skip_keys:
                continue

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        X[index] = flat_x.reshape(datum.channels, datum.height, datum.width)
        y[index] = datum.label
        keys[index] = key
        index += 1

    return X, y, keys


def criterium(uncertainty, correct, keys):

    # keys = keys[correct]
    # uncertainty = uncertainty[correct]
    threshold = 0.9
    sorted_keys = sorted(zip(uncertainty, keys), key=lambda x: x[0], reverse=True)

    for i in xrange(0, len(sorted_keys), len(sorted_keys)/10):
        print sorted_keys[i]

    sorted_keys = [x[1] for x in sorted_keys if x[0] > threshold]
    print 'Chosen {0} keys with threshold {1}'.format(len(sorted_keys), threshold)
    return sorted_keys

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
        self.first()

    def close(self):
        self.env.close()
        self.env = None

    def is_opened(self):
        return self.env is not None

    def go_to_index(self, cursor, index):
        if index > self.size:
            raise ValueError('Index out of bounds')

        cursor.first()
        while index > 0:
            index -= 1
            cursor.next()

    def first(self):
        self.index = 0

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


def choose_active(net, db_path, num_batches_to_choose, skip_keys=set()):

    print 'Choosing active samples....'

    dataset = Dataset(db_path, net.batch_size, skip_keys)

    total_num_batches = dataset.num_batches()
    total_num_samples = len(dataset)
    num_batches_to_choose = min(num_batches_to_choose, total_num_batches)

    # reserve storage for outputs
    uncert = np.zeros(total_num_samples, dtype=np.float32)
    correct = np.zeros(total_num_samples, dtype=bool)
    keys = np.zeros(total_num_samples, dtype=np.object)

    dataset.open()
    for batch_num, batch in enumerate(dataset):
        utils.wait_bar('Batch number', '', batch_num + 1, total_num_batches)
        beg = batch_num * net.batch_size
        end = beg + net.batch_size

        X, y, batch_keys = batch
        output = net.forward(X)[0]
        uncert[beg:end] = utils.entropy(utils.softmax(output))
        correct[beg:end] = np.equal(output.argmax(axis=1), y)
        keys[beg:end] = batch_keys

    dataset.close()
    print
    print 'Accuracy = {0}, mean uncertainty = {1}'.format(correct.mean(), uncert.mean())

    chosen_keys = criterium(uncert, correct, keys)

    num_to_return = min((len(chosen_keys) / net.batch_size), num_batches_to_choose) * net.batch_size
    chosen_keys = chosen_keys[:num_to_return]
    skip_keys.update(chosen_keys)

    print 'Used {0} samples. Max used sample = {1}'.format(len(skip_keys), max(skip_keys))
    print 'Returning {0} new samples'.format(len(chosen_keys))
    assert len(skip_keys) <= int(max(skip_keys)), 'Index of the highest sample is lower than the number of used samples'
    return chosen_keys




