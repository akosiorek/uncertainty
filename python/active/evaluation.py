#!/usr/bin/env python

import os
import sys
import shutil
import re
import numpy as np
import caffe
import lmdb
import samples
from active_learning import *


def evaluate(model_file, pretrained_net, mean_file, db, batch_size):

    input_shape = samples.entry_shape(db)
    size = samples.len_db(db)
    num_batches = size / batch_size

    uncertainty = np.zeros(size, dtype=np.float32)
    correct = np.zeros(size, dtype=np.int8)

    net = caffe.Net(model_file, pretrained_net, caffe.TEST)
    env = lmdb.open(db, readonly=True)

    mean = samples.read_meanfile(mean_file)

    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()

        for index in xrange(num_batches):
            print 'Evaluating batch {0}/{1}....\r'.format(index+1, num_batches),
            sys.stdout.flush()

            # print 'Running batch #{0}'.format(index)
            beg = index * batch_size
            end = (index+1) * batch_size

            X, y, keys = samples.build_batch(cursor, batch_size, input_shape)
            X -= mean

            net.forward(data=X)
            y_predicted = net.blobs["ip2"].data.argmax(axis=1)
            uncertainty[beg:end] = net.blobs["uncertainty"].data.squeeze()
            correct[beg:end] = np.equal(y, y_predicted)

    print
    return uncertainty, correct


def write_to_file(path, container):
    with open(path, 'w') as f:
        f.write('\n'.join([str(e) for e in container]))

if __name__ == '__main__':
    args = sys.argv[1:]

    net_path, db_path, snapshot_folder, results_folder = args
    batch_size, mean_file = get_batch_mean_from_net(net_path)
    input_shape = samples.entry_shape(db_path)
    deploy_net_path = net_path + '.deploy' + POSTFIX
    prepare_deploy_net(net_path, deploy_net_path, batch_size, input_shape)

    caffe.set_mode_gpu()

    files = os.listdir(snapshot_folder)
    files = [f for f in files if 'caffemodel' in f]
    print files

    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.mkdir(results_folder)

    for pretrained in files:
        print 'Processing {0}'.format(pretrained)
        pretrained = os.path.join(snapshot_folder, pretrained)
        num = re.search(r'_(\d+)', pretrained).groups()[0]
        uncert, correct = evaluate(deploy_net_path, pretrained, mean_file, db_path, batch_size)

        uncert_path = os.path.join(results_folder, 'uncert_{0}.txt'.format(num))
        label_path = os.path.join(results_folder, 'label_{0}.txt'.format(num))

        write_to_file(uncert_path, uncert)
        write_to_file(label_path, correct)
