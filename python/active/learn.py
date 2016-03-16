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

import config

import os
import sys
import shutil
import subprocess

import caffe

import db
import samples
import proto
import utils
import net


def init_network(solver_path):
    """# 1. Initialize the net by training with an initial solver"""
    print 'Initializing net..."'
    cmd = '{0} train --solver={1}'.format(config.CAFFE_EXEC, solver_path)
    with open('log.txt', 'w') as f:
        subprocess.call(cmd.split(' '), stdout=f, stderr=subprocess.STDOUT)


def train_network(solver_path, snapshot_path):
    print 'Training net..."'
    cmd = '{0} train --solver={1} --snapshot={2}'.format(config.CAFFE_EXEC, solver_path, snapshot_path)
    with open('log.txt', 'a') as f:
        subprocess.call(cmd.split(' '), stdout=f, stderr=subprocess.STDOUT)


def learn(solver_path, snapshot_path, iters_to_init, max_samples_to_use):
    net_path = proto.get_net_from_solver(solver_path)
    train_db_path = proto.get_db_from_net(net_path)
    train_db_len = db.size(train_db_path)

# prepare path to the temporary model files
    active_solver_path = utils.create_temp_path(solver_path + config.POSTFIX)
    active_net_path = utils.create_temp_path(net_path + config.POSTFIX)
    active_db_path = utils.create_temp_path(train_db_path + config.POSTFIX)

# prepare temporary model files
    proto.prepare_net(net_path, active_net_path, active_db_path)
    snapshot_prefix, snapshot_iter = proto.prepare_solver(
        solver_path, active_solver_path, active_net_path, snapshot_path, iters_to_init
    )

    print snapshot_prefix

    # recover the snapshot folder
    snapshot_path = '/'.join(snapshot_prefix.split('/')[:-1])
    epoch_file = os.path.join(snapshot_path, config.EPOCH_FILE)

    # deploy net
    # deploy_net = net.Net(active_net_path, output_layers=config.OUTPUT_LAYERS)
    deploy_net = net.DropoutNet(active_net_path, config.DROPOUT_ITERS, aggregate='mean', output_layers=config.OUTPUT_LAYERS)

    epoch_used_samples = set()
    dataset = samples.Dataset(train_db_path, deploy_net.batch_size, epoch_used_samples)

# initialize net
#     solverstate_path = proto.solverstate_path(snapshot_prefix, iters_to_init)
#     if not os.path.exists(solverstate_path):
    if os.path.exists(active_db_path):
        shutil.rmtree(active_db_path)
        # shutil.copytree(train_db_path, active_db_path)

    used_samples = db.extract_samples(train_db_path, active_db_path, iters_to_init * deploy_net.batch_size)
    init_network(active_solver_path)

# do the real learning
    print 'train samples:', train_db_len
    for epoch in xrange(config.MAX_EPOCHS):
        print 'Epoch #{0}'.format(epoch)
        epoch_used_samples.clear()
        epoch_used_samples.update(used_samples)

        while len(epoch_used_samples) < train_db_len:
            if snapshot_iter > config.MAX_ITER:
                break

            solverstate_path = proto.solverstate_path(snapshot_prefix, snapshot_iter)
            caffemodel_path = proto.caffemodel_path(snapshot_prefix, snapshot_iter)

            print 'Using snapshot iter #{0}'.format(snapshot_iter)
            deploy_net.load_model(caffemodel_path)
            # active_samples = samples.choose_active(deploy_net, dataset, config.BATCHES_PER_RUN)
            # epoch_used_samples.update(active_samples)
            # assert len(active_samples) <= int(max(active_samples)), \
            #     'Index of the highest sample is lower than the number of used samples'
            #            # check if it makes sense to continue
            # iters_to_do = len(active_samples) / deploy_net.batch_size
            # if iters_to_do == 0:
            #     break

            num_samples_to_choose = min(max_samples_to_use - len(epoch_used_samples), config.NEW_SAMPLES_PER_ITER)
            batches_to_choose = num_samples_to_choose / deploy_net.batch_size

            chosen_samples = samples.choose_active(deploy_net, dataset, batches_to_choose)
            active_samples = chosen_samples + list(epoch_used_samples)
            epoch_used_samples.update(chosen_samples)

            print 'Used {} samples'.format(len(epoch_used_samples))
            # check if it makes sense to continue
            iters_to_do = len(active_samples) / deploy_net.batch_size
            if iters_to_do == 0:
                break

            db.extract_samples(train_db_path, active_db_path, active_samples)
            proto.increase_max_iters(active_solver_path, iters_to_do)
            train_network(active_solver_path, solverstate_path)

            snapshot_iter += iters_to_do
            utils.append_to_file(epoch_file, '{}:{}'.format(snapshot_iter, len(epoch_used_samples)))


if __name__ == '__main__':
    args = sys.argv[1:]
    solver_path = args[0]

    snapshot_path = None
    iters_to_init = config.ITERS_TO_INIT
    max_samples_to_use = None

    if len(args) > 1:
        snapshot_path = args[1]

    if len(args) > 2:
        iters_to_init = int(args[2])

    if len(args) > 3:
        max_samples_to_use = int(args[3])

    caffe.set_mode_gpu()
    add_per_iter = [200, 300, 400, 500, 1000, 2500]

    if not os.path.exists(snapshot_path):
        os.mkdir(snapshot_path)

    for iters in add_per_iter:
        snapshot_folder = os.path.join(snapshot_path, str(iters))
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)

        config.NEW_SAMPLES_PER_ITER = iters
        learn(solver_path, snapshot_folder, iters_to_init, max_samples_to_use)
