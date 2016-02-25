#!/usr/bin/env python
"""Performs Active Learning

1. Initialize the network
  * Run normal training for N_0 iterations
2. For iter = 1:max_iter
  1) Evaluate uncertainty on training test
  * A Python script that takes a deploy.prototxt and evaluates uncertainty and correctness of samples from the training set; stores appropriate labels and samples in a new db
  2) Pick k minibatches of certain/incorrect and uncertain/correct samples
  * Possibly use a different solver; Switch the db with the newly created db
  3) Train NN
  * Run only as long as there is unused data
  4) ++iter and go to 1)


To do that we need:
  1. Solver file, in which we will increase the number of iterations
  2. Network file - just one, the same throughout training
  3. Script that will evaluate uncertainty of samples from a DB
  4. Another script that will extract samples from a db and put them in a different db
  



"""

import random
import os
import shutil
import numpy as np
import subprocess
import lmdb

import caffe

import samples
from proto import *


CAFFE_EXEC = '../build/tools/caffe'
UNCERTAINTY_LAYER = 'uncertainty'
UNCERTAINTY_DB = 'uncertainty_lmdb'
MODEL_FOLDER = 'models/uncertainty'
SOLVER = 'solver.prototxt'
POSTFIX = '.active'

BATCHES_PER_RUN = 100

TRAIN_PHASE = 0
TEST_PHASE = 1


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


# def criterium(uncertainty, correct, keys):
#     index = np.abs(correct - uncertainty) < 0.5
#     return keys[index]


mean_uncert = 0.5


def criterium(uncertainty, correct, keys):
    score = np.abs(correct - uncertainty)
    current_mean = score.mean()
    global mean_uncert
    mean_uncert = 0.999 * mean_uncert + 0.001 * current_mean
    print mean_uncert, current_mean
    return keys[score > 1.25 * mean_uncert]


if __name__ == '__main__':
    args = sys.argv[1:]

    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()
    max_epochs = 10
    solver_path = args[0]

    # solver_path = os.path.join(MODEL_FOLDER, solver_path)
    net_path = get_net_from_solver(solver_path)
    train_db_path = get_db_from_net(net_path)

    active_solver_path = solver_path + POSTFIX
    active_net_path = net_path + POSTFIX
    active_db_path = train_db_path + POSTFIX
    deploy_net_path = net_path + '.deploy' + POSTFIX

    # TODO: remove; for now needed since the active db does not exist yet and init doesn't work
    prepare_net(net_path, active_net_path, train_db_path)
    snapshot_prefix, snapshot_iter = prepare_solver(solver_path, active_solver_path, active_net_path)

    # init_network(active_solver_path)

    batch_size, mean_file = get_batch_mean_from_net(net_path)
    prepare_net(net_path, active_net_path, active_db_path)

    # deploy net
    input_shape = samples.entry_shape(train_db_path)
    prepare_deploy_net(net_path, deploy_net_path, batch_size, input_shape)

    train_db_len = samples.len_db(train_db_path)
    print 'train samples:', train_db_len
    for epoch in xrange(max_epochs):
        print 'Epoch #{0}'.format(epoch)
        used_samples = set()
        iters_to_do = 1

        while len(used_samples) < train_db_len:

            solverstate_path = '{0}_iter_{1}.solverstate.h5'.format(snapshot_prefix, snapshot_iter)
            caffemodel_path = '{0}_iter_{1}.caffemodel.h5'.format(snapshot_prefix, snapshot_iter)
            active_samples = samples.choose_active(
                    deploy_net_path,
                    caffemodel_path, mean_file, train_db_path,
                    batch_size, criterium, BATCHES_PER_RUN, train_db_len/batch_size, input_shape, used_samples
            )

            iters_to_do = min(len(active_samples) / batch_size, BATCHES_PER_RUN)

            # check if it makes sense to continue
            if iters_to_do == 0:
                break

            create_db_with_samples(train_db_path, active_db_path, active_samples)
            increase_max_iters(active_solver_path, iters_to_do)
            train_network(active_solver_path, solverstate_path)
            snapshot_iter += iters_to_do

            print len(used_samples)

































