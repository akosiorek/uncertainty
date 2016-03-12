import math
import numpy as np
import caffe
import lmdb

import utils
import db


DROPOUT_ITERS = 10
OUTPUT_LAYER = 'ip2'
SAMPLE_MEAN = None


def read_meanfile(mean_path):
    print 'Reading mean file...'
    data = open(mean_path , 'rb').read()
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    return arr[0]


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


def uncertainty_dropout_argmax(X, y, net):
    probs = np.zeros(net.blobs[OUTPUT_LAYER].data.shape, dtype=np.float32)
    lin_index = range(probs.shape[0])
    for _ in xrange(DROPOUT_ITERS):
        net.forward(data=X)
        argmax = net.blobs[OUTPUT_LAYER].data.argmax(axis=1)
        probs[lin_index, argmax] += 1
    probs /= DROPOUT_ITERS

    uncert = utils.entropy(probs)
    correct = np.equal(y, probs.argmax(axis=1))
    return uncert, correct


def uncertainty_dropout_sum(X, y, net):
    probs = np.zeros(net.blobs[OUTPUT_LAYER].data.shape, dtype=np.float32)
    for _ in xrange(DROPOUT_ITERS):
        net.forward(data=X)
        probs += utils.softmax(net.blobs[OUTPUT_LAYER].data)
    probs /= DROPOUT_ITERS

    uncert = utils.entropy(probs)
    correct = np.equal(y, probs.argmax(axis=1))
    return uncert, correct


def uncertainty_base_case(X, y, net):
    net.forward(data=X)
    y_predicted = net.blobs[OUTPUT_LAYER].data.argmax(axis=1)
    uncert = utils.entropy(utils.softmax(net.blobs[OUTPUT_LAYER].data))
    correct = np.equal(y, y_predicted)
    return uncert, correct


def criterium(uncertainty, correct, keys):

    keys = keys[correct]
    uncertainty = uncertainty[correct]
    threshold = 0.3
    sorted_keys = sorted(zip(uncertainty, keys), key=lambda x: x[0], reverse=True)

    for i in xrange(0, len(sorted_keys), len(sorted_keys)/10):
        print sorted_keys[i]

    sorted_keys = [x[1] for x in sorted_keys if x[0] > threshold]
    print 'Chosen {0} keys with threshold {1}'.format(len(sorted_keys), threshold)
    return sorted_keys


class SamplePool(object):

    def __init__(self):
        pass


class Policy(object):
    pass

class Scorer(object):
    pass


def setup():

    db_path = ''
    scorer = Scorer()
    policy = Policy()

    pool = SamplePool(db_path, scorer, policy)

    epochs = 10

    for e in xrange(epochs):
        # run epoch
        for batch in pool:
            pass

        SamplePool.reset()




def choose_active(model_file, pretrained_net, mean_file, db, batch_size, num_batches_to_choose, total_num_batches,
                  input_shape, skip_keys=set(), used_samples=set(), max_samples_to_use=-1):

    print 'Choosing active samples....'

    total_num_samples = total_num_batches * batch_size - len(skip_keys)
    total_num_batches = int(math.ceil(float(total_num_samples) / batch_size))
    total_num_samples = total_num_batches * batch_size

    num_batches_to_choose = min(num_batches_to_choose, total_num_batches)
    net = caffe.Net(model_file, pretrained_net, caffe.TEST)
    env = lmdb.open(db, readonly=True)

    uncert = np.zeros(total_num_samples, dtype=np.float32)
    correct = np.zeros(total_num_samples, dtype=bool)
    keys = np.zeros(total_num_samples, dtype=np.object)

    global SAMPLE_MEAN
    if SAMPLE_MEAN is None:
        SAMPLE_MEAN = read_meanfile(mean_file)

    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()

        for batch_num in xrange(total_num_batches):
            utils.wait_bar('Batch number', '', batch_num + 1, total_num_batches)
            beg = batch_num * batch_size
            end = beg + batch_size

            batch = build_batch(cursor, batch_size, input_shape, skip_keys)

            if batch is None:
                uncert = uncert[:beg]
                correct = correct[:beg]
                keys = keys[:beg]
                print 'Couldn\'t choose a batch'
                break

            X, y, batch_keys = batch
            X -= SAMPLE_MEAN

            uncert[beg:end], correct[beg:end] = uncertainty_dropout_sum(X, y, net)
            keys[beg:end] = batch_keys

    env.close()
    print
    print 'Accuracy = {0}, mean uncertainty = {1}'.format(correct.mean(), uncert.mean())

    chosen_keys = criterium(uncert, correct, keys)

    num_to_return = min((len(chosen_keys) / batch_size), num_batches_to_choose) * batch_size
    chosen_keys = chosen_keys[:num_to_return]
    skip_keys.update(chosen_keys)

    print 'Used {0} samples. Max used sample = {1}'.format(len(skip_keys), max(skip_keys))
    print 'Returning {0} new samples'.format(len(chosen_keys))
    assert len(skip_keys) <= int(max(skip_keys)), 'Indext of the highest sample is lower than the number of used samples'
    return chosen_keys




