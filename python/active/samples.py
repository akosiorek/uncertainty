
import math
import numpy as np
import caffe
import lmdb

import utils

def entry_shape(db_path):

    env = lmdb.open(db_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()

        cursor.first()
        key, raw_datum = cursor.item()
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        shape = datum.channels, datum.height, datum.width
    env.close()
    return shape


def len_db(db_path):
    env = lmdb.open(db_path, readonly=True)
    size = 0
    with env.begin() as txn:
        for _ in txn.cursor():
            size += 1
    env.close()
    return size


def read_meanfile(mean_path):
    print 'Reading mean file...'
    data = open(mean_path , 'rb').read()
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    return arr[0]


def move_cursor_circular(cursor):
    moved = cursor.next()
    if not moved:
        cursor.first()


def build_batch(cursor, batch_size, input_shape, skip_keys=None):

    X = np.zeros([batch_size] + list(input_shape), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.int32)
    keys = np.zeros(batch_size, dtype=np.object)

    first_key = None
    index = 0
    while index < batch_size:
        key, raw_datum = cursor.item()
        move_cursor_circular(cursor)

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


sample_mean = None


def choose_active(model_file, pretrained_net, mean_file, db, batch_size, num_batches_to_choose, total_num_batches,
                  input_shape, skip_keys=set(), criterium=None):

    print 'Choosing active samples....'

    total_num_samples = total_num_batches * batch_size - len(skip_keys)
    total_num_batches = int(math.ceil(float(total_num_samples) / batch_size))

    num_to_choose = min(num_batches_to_choose, total_num_batches) * batch_size
    chosen_keys = []
    net = caffe.Net(model_file, pretrained_net, caffe.TEST)
    env = lmdb.open(db, readonly=True)

    global sample_mean
    if sample_mean is None:
        sample_mean = read_meanfile(mean_file)

    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        key, _ = cursor.item()

        for batch_num in xrange(total_num_batches):
            batch = build_batch(cursor, batch_size, input_shape, skip_keys)

            if batch is None:
                print 'Couldn\'t choose a batch'
                break

            X, y, keys = batch
            X -= sample_mean

            net.forward(data=X)
            y_predicted = net.blobs["ip2"].data.argmax(axis=1)
            uncertainty = net.blobs["uncertainty"].data.squeeze()
            correct = np.equal(y, y_predicted)

            keys_to_add = criterium(uncertainty, correct, keys)
            num_to_add = min(len(keys_to_add), num_to_choose-len(chosen_keys))

            keys_to_add = keys_to_add[:num_to_add]

            chosen_keys.extend(keys_to_add)
            skip_keys.update(keys_to_add)

            utils.wait_bar('Batch number={0}, Chosen'.format(batch_num), 'keys', len(chosen_keys), num_to_choose)
            if len(chosen_keys) == num_to_choose:
                break

    print
    env.close()

    num_to_return = (len(chosen_keys) / batch_size) * batch_size
    save_for_later = chosen_keys[num_to_return:]
    print len(skip_keys)
    skip_keys.difference_update(save_for_later)
    print len(skip_keys)
    print len(set(chosen_keys))
    chosen_keys = chosen_keys[:num_to_return]

    print 'Used {0} samples. Max used sample = {1}'.format(len(skip_keys), max(skip_keys))
    print 'Returning {0} new samples'.format(len(chosen_keys))
    assert len(skip_keys) <= int(max(skip_keys)), 'Indext of the highest sample is lower than the number of used samples'
    return chosen_keys




