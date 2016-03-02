#!/usr/bin/env python

import os
import sys
import shutil
import numpy as np

import lmdb

os.environ['GLOG_minloglevel'] = '1'
import caffe

import active_learning
import utils
import samples
import proto

UNCERTS = ['max_unc', 'entropy_conf', 'entropy_ip2', '2_max_ip2', 'entropy_weighted', '2_max_weighted']
NUM_UNC = len(UNCERTS)


def evaluate(model_file, pretrained_net, db, batch_size, mean):

    input_shape = samples.entry_shape(db)
    size = samples.len_db(db)
    num_batches = size / batch_size

    uncertainty = np.zeros((size, NUM_UNC), dtype=np.float32)
    correct = np.zeros(size, dtype=np.int8)

    net = caffe.Net(model_file, pretrained_net, caffe.TEST)
    env = lmdb.open(db, readonly=True)

    print 'Processing {0}'.format(pretrained_net)
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()

        for index in xrange(num_batches):
            print 'Evaluating batch {0}/{1}...\r'.format(index+1, num_batches),
            sys.stdout.flush()

            # print 'Running batch #{0}'.format(index)
            beg = index * batch_size
            end = (index+1) * batch_size

            X, y, keys = samples.build_batch(cursor, batch_size, input_shape)
            X -= mean

            net.forward(data=X)

            weighted = utils.softmax(net.blobs["weighted_input"].data)
            ip2 = utils.softmax(net.blobs["ip2"].data)
            confidence = net.blobs["confidence"].data

            y_predicted = weighted.argmax(axis=1)

            uncertainty[beg:end, 0] = 1-confidence[xrange(y_predicted.shape[0]), y_predicted]
            uncertainty[beg:end, 1] = utils.entropy(confidence)
            uncertainty[beg:end, 2] = utils.entropy(ip2)
            uncertainty[beg:end, 3] = utils.second_max(ip2)
            uncertainty[beg:end, 4] = utils.entropy(weighted)
            uncertainty[beg:end, 5] = utils.second_max(weighted)

            correct[beg:end] = np.equal(y, y_predicted)

    print
    return uncertainty, correct


if __name__ == '__main__':
    args = sys.argv[1:]

    net_path, db_path, snapshot_folder, results_folder = args

    batch_size, mean_file = proto.get_batch_mean_from_net(net_path)
    input_shape = samples.entry_shape(db_path)
    deploy_net_path = net_path + '.deploy' + active_learning.POSTFIX
    proto.prepare_deploy_net(net_path, deploy_net_path, batch_size, input_shape)

    caffe.set_mode_gpu()
    mean = samples.read_meanfile(mean_file)

    files = utils.get_snapshot_files(snapshot_folder)

    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)

    output_folders = [os.path.join(results_folder, output_name) for output_name in UNCERTS]

    os.mkdir(results_folder)
    for output_folder in output_folders:
        os.mkdir(output_folder)

    for snapshot_num, pretrained in files:
        print 'Processing {0}'.format(pretrained)
        pretrained = os.path.join(snapshot_folder, pretrained)
        num = utils.get_snapshot_number(pretrained)
        uncert, correct = evaluate(deploy_net_path, pretrained, db_path, batch_size, mean)

        for output_num, outut_folder in enumerate(output_folders):

            uncert_path = os.path.join(outut_folder, 'uncert_{0}.txt'.format(num))
            label_path = os.path.join(outut_folder, 'label_{0}.txt'.format(num))

            utils.write_to_file(uncert_path, uncert[:, output_num])
            utils.write_to_file(label_path, correct)
