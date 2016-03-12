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
import subprocess

os.environ['GLOG_minloglevel'] = '1'
import caffe

import db
import samples
import proto
import utils

CAFFE_EXEC = '../build/tools/caffe'
MODEL_FOLDER = 'models/uncertainty'
POSTFIX = '.active'
EPOCH_FILE = 'epochs.txt'

MAX_EPOCHS = 10
BATCHES_PER_RUN = 100
ITERS_TO_INIT = 1000


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


def learn(solver_path, snapshot_path, iters_to_init, max_samples_to_use):
    net_path = proto.get_net_from_solver(solver_path)
    train_db_path = proto.get_db_from_net(net_path)

# setup sample budgeting
    batch_size, mean_file = proto.get_batch_mean_from_net(net_path)
    train_db_len = db.size(train_db_path)
    total_num_batches = train_db_len / batch_size
    max_samples_to_use = min(train_db_len, max_samples_to_use)
    num_used_samples = iters_to_init * batch_size

    if num_used_samples > max_samples_to_use:
        iters_to_init = max_samples_to_use / batch_size
        num_used_samples = iters_to_init * batch_size

# prepare path to the temporary model files
    active_solver_path = utils.create_temp_path(solver_path + POSTFIX)
    active_net_path = utils.create_temp_path(net_path + POSTFIX)
    active_db_path = utils.create_temp_path(train_db_path + POSTFIX)
    deploy_net_path = utils.create_temp_path(net_path + '.deploy' + POSTFIX)

# prepare temporary model files
    proto.prepare_net(net_path, active_net_path, active_db_path)
    snapshot_prefix, snapshot_iter = proto.prepare_solver(
        solver_path, active_solver_path, active_net_path, snapshot_path, iters_to_init
    )

    # recover the snapshot folder
    snapshot_path = '/'.join(snapshot_prefix.split('/')[:-1])
    epoch_file = os.path.join(snapshot_path, EPOCH_FILE)

    # deploy net
    input_shape = db.entry_shape(train_db_path)
    proto.prepare_deploy_net(net_path, deploy_net_path, batch_size, input_shape)

# initialize net
    used_samples = set()
    solverstate_path = proto.solverstate_path(snapshot_prefix, iters_to_init)
    if not os.path.exists(solverstate_path):
        used_samples = db.extract_samples(train_db_path, active_db_path, num_used_samples)
        init_network(active_solver_path)

# do the real learning
    print 'train samples:', train_db_len
    for epoch in xrange(MAX_EPOCHS):
        print 'Epoch #{0}'.format(epoch)
        epoch_used_samples = set()

        while len(epoch_used_samples) < train_db_len:

            solverstate_path = proto.solverstate_path(snapshot_prefix, snapshot_iter)
            caffemodel_path = proto.caffemodel_path(snapshot_prefix, snapshot_iter)

            print 'Using snapshot iter #{0}'.format(snapshot_iter)

            active_samples = samples.choose_active(
                    deploy_net_path, caffemodel_path, mean_file, train_db_path,
                    batch_size, BATCHES_PER_RUN, total_num_batches, input_shape, epoch_used_samples,
                    used_samples, max_samples_to_use
            )

            # check if it makes sense to continue
            iters_to_do = len(active_samples) / batch_size
            if iters_to_do == 0:
                break

            db.extract_samples(train_db_path, active_db_path, active_samples)
            proto.increase_max_iters(active_solver_path, iters_to_do)
            train_network(active_solver_path, solverstate_path)
            snapshot_iter += iters_to_do

        utils.append_to_file(epoch_file, str(snapshot_iter))


if __name__ == '__main__':
    args = sys.argv[1:]
    solver_path = args[0]

    snapshot_path = None
    iters_to_init = ITERS_TO_INIT
    max_samples_to_use = sys.maxint

    if len(args) > 1:
        snapshot_path = args[1]
        if not os.path.exists(snapshot_path):
            os.mkdir(snapshot_path)

    if len(args) > 2:
        iters_to_init = int(args[2])

    if len(args) > 3:
        max_samples_to_use = int(args[3])


    caffe.set_mode_gpu()
    learn(solver_path, snapshot_path, iters_to_init, max_samples_to_use)


































