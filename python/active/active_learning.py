#!/usr/bin/env python
"""Performs Active Learning

1. Initialize the network
  * Run normal training for N_0 iterations
2. For iter = 1:max_iter
  1) Evaluate uncertainty on the training set
  2) Pick k minibatches of certain/incorrect and uncertain/correct samples
  3) Train NN
  4) ++iter and go to 1)
"""

import os
import sys
import random
import shutil
import subprocess

import lmdb

os.environ['GLOG_minloglevel'] = '1'
import caffe

import samples
import proto
import utils

CAFFE_EXEC = '../build/tools/caffe'
MODEL_FOLDER = 'models/uncertainty'
POSTFIX = '.active'

MAX_EPOCHS = 10
BATCHES_PER_RUN = 100


def init_network(solver_path):
    """# 1. Initialize the net by training with an initial solver"""
    print 'Initializing net..."'
    cmd = '{0} train --solver={1}'.format(CAFFE_EXEC, solver_path)
    with open('log.txt', 'w') as f:
        subprocess.call(cmd.split(' '), stdout=f, stderr=subprocess.STDOUT)


def train_network(solver_path, snapshot_path):
    print 'Training net..."'
    cmd = '{0} train --solver={1} --snapshot={2}'.format(CAFFE_EXEC, solver_path, snapshot_path)
    with open('log.txt', 'a') as f:
        subprocess.call(cmd.split(' '), stdout=f, stderr=subprocess.STDOUT)


def create_db_with_samples(input_db, output_db, sample_keys):

    if os.path.exists(output_db):
        shutil.rmtree(output_db)

    map_size=2**40
    env_in = lmdb.open(input_db, readonly=True, map_size=map_size)
    env_out = lmdb.open(output_db, readonly=False, map_size=map_size)

    sample_keys = sorted(sample_keys)
    ids = range(len(sample_keys))
    random.shuffle(ids)

    with env_in.begin() as txn_in:
        cursor_in = txn_in.cursor()
        with env_out.begin(write=True) as txn_out:
            for id, key in zip(ids, sample_keys):
                str_id = '{:08}'.format(id)
                txn_out.put(str_id, cursor_in.get(key))

    env_in.close()
    env_out.close()


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


if __name__ == '__main__':
    args = sys.argv[1:]
    solver_path = args[0]

    if len(args) > 1:
        snapshot_path = args[1]
        utils.clear_dir(snapshot_path)
    else:
        snapshot_path = None

    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()

    net_path = proto.get_net_from_solver(solver_path)
    train_db_path = proto.get_db_from_net(net_path)

    active_solver_path = utils.create_temp_path(solver_path + POSTFIX)
    active_net_path = utils.create_temp_path(net_path + POSTFIX)
    active_db_path = utils.create_temp_path(train_db_path + POSTFIX)
    deploy_net_path = utils.create_temp_path(net_path + '.deploy' + POSTFIX)

    # TODO: remove; for now needed since the active db does not exist yet and init doesn't work
    proto.prepare_net(net_path, active_net_path, train_db_path)
    snapshot_prefix, snapshot_iter = proto.prepare_solver(solver_path, active_solver_path, active_net_path, snapshot_path)

    init_network(active_solver_path)

    batch_size, mean_file = proto.get_batch_mean_from_net(net_path)
    proto.prepare_net(net_path, active_net_path, active_db_path)

    # deploy net
    input_shape = samples.entry_shape(train_db_path)
    proto.prepare_deploy_net(net_path, deploy_net_path, batch_size, input_shape)

    train_db_len = samples.len_db(train_db_path)
    print 'train samples:', train_db_len
    for epoch in xrange(MAX_EPOCHS):
        print 'Epoch #{0}'.format(epoch)
        used_samples = set()
        iters_to_do = 1

        while len(used_samples) < train_db_len:

            solverstate_path = '{0}_iter_{1}.solverstate.h5'.format(snapshot_prefix, snapshot_iter)
            caffemodel_path = '{0}_iter_{1}.caffemodel.h5'.format(snapshot_prefix, snapshot_iter)
            active_samples = samples.choose_active(
                    deploy_net_path, caffemodel_path, mean_file, train_db_path,
                    batch_size, BATCHES_PER_RUN, train_db_len/batch_size, input_shape, used_samples, criterium
            )

            iters_to_do = min(len(active_samples) / batch_size, BATCHES_PER_RUN)

            # check if it makes sense to continue
            if iters_to_do == 0:
                break

            create_db_with_samples(train_db_path, active_db_path, active_samples)
            proto.increase_max_iters(active_solver_path, iters_to_do)
            train_network(active_solver_path, solverstate_path)
            snapshot_iter += iters_to_do
































