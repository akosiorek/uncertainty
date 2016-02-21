#!/usr/bin/env python

import os
import sys
import shutil
import re
import numpy as np
import caffe
import lmdb
import samples


def evaluate(model_file, pretrained_net, mean_file, db, batch_size):
    print 'Evaluating....'

    input_shape = samples.entry_shape(db)
    size = samples.len_db(db)
    num_batches = size / batch_size

    uncertainty = np.zeros(size, dtype=np.float32)
    correct = np.zeros(size, dtype=np.int8)

    net = caffe.Net(model_file, pretrained_net, caffe.TEST)
    env = lmdb.open(db, readonly=True)

    mean = samples.read_meanfile(mean_file)

    index = 0
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        key, _ = cursor.item()
        first_key = None

        while index < batch_size:
            beg = index * batch_size
            end = (index+1) * batch_size

            batch = samples.build_batch(cursor, batch_size, input_shape, first_key)
            if first_key is None:
                first_key = key

            if batch is None:
                break

            X, y, keys = batch
            X -= mean

            net.forward(data=X)
            y_predicted = net.blobs["ip2"].data.argmax(axis=1)
            uncertainty[beg:end] = net.blobs["uncertainty"].data.squeeze()
            correct[beg:end] = np.equal(y, y_predicted)
            index += 1
            
    return uncertainty, correct


def write_to_file(path, container):
    with open(path, 'w') as f:
        f.write('\n'.join([str(e) for e in container]))

if __name__ == '__main__':
    args = sys.argv[1:]

    caffe.set_mode_gpu()

    model_file = args[0]
    mean_file = args[1]
    batch_size = int(args[2])
    db_path = args[3]
    input_folder = args[4]
    output_folder = args[5]

    files = os.listdir(input_folder)
    files = [f for f in files if 'caffemodel' in f]
    print files

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    for pretrained in files:
        print 'Processing {0}'.format(pretrained)
        pretrained = os.path.join(input_folder, pretrained)
        num = re.search(r'_(\d+)', pretrained).groups()[0]
        uncert, correct = evaluate(model_file, pretrained, mean_file, db_path, batch_size)

        uncert_path = os.path.join(output_folder, 'uncert_{0}.txt'.format(num))
        label_path = os.path.join(output_folder, 'label_{0}.txt'.format(num))

        write_to_file(uncert_path, uncert)
        write_to_file(label_path, correct)
