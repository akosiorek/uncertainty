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
import sys
import shutil
import numpy as np
import subprocess
import lmdb
from google.protobuf import text_format
import caffe
from caffe.proto.caffe_pb2 import SolverParameter, NetParameter, BlobShape
import samples

CAFFE_EXEC = '../build/tools/caffe'
FEATURES_EXEC = '../build/tools/extract_features'
UNCERTAINTY_LAYER = 'uncertainty'
UNCERTAINTY_DB = 'uncertainty_lmdb'
MODEL_FOLDER = 'models/uncertainty'
SOLVER = 'solver.prototxt'
POSTFIX = '.active'
DEPLOY_NET = 'deploy.prototxt'

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


def extract_features(snapshot_prefix, iter, net, batches):
    cmd = '{0} {1}_iter_{2}.caffemodel.h5 net.txt {3} {4} {5} lmdb GPU'.format(
            FEATURES_EXEC, snapshot_prefix, iter, UNCERTAINTY_LAYER, UNCERTAINTY_DB, batches
    )
    subprocess.call(cmd.split())


def load_proto(path, proto_type):
    proto = proto_type()
    data = open(path).read()
    text_format.Merge(data, proto)
    return proto


def save_proto(path, proto):
    data = text_format.MessageToString(proto)
    with open(path, 'w') as f:
        f.write(data)


def get_net_from_solver(solver_path):
    proto = load_proto(solver_path, SolverParameter)
    return str(proto.net)


def get_db_from_net(net_path, phase=TRAIN_PHASE):
    proto = load_proto(net_path, NetParameter)
    for layer in proto.layer:
        if layer.type == 'Data' and layer.include[0].phase == phase:
            return layer.data_param.source


def write_db_to_net(net_proto, db, phase=caffe.TRAIN):
    for layer in net_proto.layer:
        if layer.type == 'Data' and layer.include[0].phase == phase:
            layer.data_param.source = db
            break


def prepare_solver(solver_path, out_path, prepared_net):
    print 'Preparing solver.prototxt...'
    proto = load_proto(solver_path, SolverParameter)

    proto.snapshot = 0
    proto.snapshot_after_train = True
    proto.max_iter = 2500
    proto.net = prepared_net
    proto.test_initialization = False

    save_proto(out_path, proto)
    return proto.snapshot_prefix, proto.max_iter


def prepare_net(net_path, out_path, new_train_db):
    print 'Preparing net.prototxt'
    proto = load_proto(net_path, NetParameter)

    write_db_to_net(proto, new_train_db, caffe.TRAIN)
    # write_db_to_net(proto, new_train_db, caffe.TEST)

    # # remove test layers
    # for i in xrange(len(proto.layer)-1, -1, -1):
    #     if len(proto.layer[i].include) and proto.layer[i].include[0].phase == caffe.TEST:
    #         del proto.layer[i]

    save_proto(out_path, proto)
    batch_size = proto.layer[0].data_param.batch_size
    mean_file = proto.layer[0].transform_param.mean_file
    return batch_size, mean_file


def prepare_deploy_net(net_path, out_path, batch_size, input_size):
    print 'Preparing deploy net'
    proto = load_proto(net_path, NetParameter)

    tops = set()
    for layer in proto.layer:
        if layer.type == 'Data':
            tops.update(layer.top)

    # find layers using labels
    layers_to_remove = []
    for i in xrange(len(proto.layer)):
        for b in proto.layer[i].bottom:
            if b == 'label':
                layers_to_remove.append(i)
                break


    input_shape = BlobShape()
    input_shape.dim.extend([batch_size])
    input_shape.dim.extend(input_size)

    proto.input.extend([proto.layer[0].top[0]])
    proto.input_shape.extend([input_shape])

    # remove layers using labels
    for i in reversed(layers_to_remove):
        del proto.layer[i]

    # remove data layers
    del proto.layer[1]
    del proto.layer[0]

    save_proto(out_path, proto)

# def choose_active_samples(net_path, snapshot_prefix, snapshot_iter, batch_size, used_samples):
#     """# 2. Evaluate uncertaintities and put appropriate samples/labels in a new db"""
#     print 'Choosing active samples...'
#
#     if os.path.exists(UNCERTAINTY_DB):
#         shutil.rmtree(UNCERTAINTY_DB)
#
#     extract_features(snapshot_prefix, snapshot_iter, net_path, 500)
#
#     scores = []
#     env = lmdb.open(UNCERTAINTY_DB, readonly=True)
#     with env.begin() as txn:
#         cursor = txn.cursor()
#
#         for key, raw_datum in cursor:
#
#             datum = Datum()
#             datum.ParseFromString(raw_datum)
#
#             scores.append((key, datum.float_data[0]))
#
#     env.close()
#     shutil.rmtree(UNCERTAINTY_DB)
#
#     # remove used samples
#     scores = [e for e in scores if e[0] not in used_samples]
#
#     # select the least certain samples
#     scores = sorted(scores, key=lambda x: x[1])
#     indices = [e[0] for e in scores if e[1] < 0.3]
#
#     iters_to_do = min(len(indices) / batch_size, BATCHES_PER_RUN)
#     indices = indices[:iters_to_do * batch_size]
#
#     used_samples.update(indices)
#     return indices, iters_to_do


def create_db_with_samples(input_db, output_db, sample_keys):

    if os.path.exists(output_db):
        shutil.rmtree(output_db)

    map_size=2**40
    env_in = lmdb.open(input_db, readonly=True, map_size=map_size)
    env_out = lmdb.open(output_db, readonly=False, map_size=map_size)

    with env_in.begin() as txn_in:
        cursor_in = txn_in.cursor()
        with env_out.begin(write=True) as txn_out:

            random.shuffle(sample_keys)
            for key in sample_keys:
                txn_out.put(key, cursor_in.get(key))

    env_in.close()
    env_out.close()


def criterium(uncertainty, correct, keys):
    index = np.abs(correct - uncertainty) < 0.5
    return keys[index]


def increase_max_iters(solver_path, how_many):
    proto = load_proto(solver_path, SolverParameter)

    proto.max_iter += how_many
    save_proto(solver_path, proto)


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
    batch_size, mean_file = prepare_net(net_path, active_net_path, active_db_path)

    # deploy net
    input_shape = samples.entry_shape(train_db_path)
    prepare_deploy_net(net_path, deploy_net_path, batch_size, input_shape)

    train_db_len = samples.len_db(train_db_path)
    print 'train samples:', train_db_len

    # init_network(active_solver_path)
    for epoch in xrange(max_epochs):
        print 'Epoch #{0}'.format(epoch)
        used_samples = set()
        iters_to_do = 1

        while len(used_samples) < train_db_len:
            # # active_samples, iters_to_do = choose_active_samples(
            # active_net_path, snapshot_prefix, 0, batch_size, used_samples
            # )

            solverstate_path = '{0}_iter_{1}.solverstate.h5'.format(snapshot_prefix, snapshot_iter)
            caffemodel_path = '{0}_iter_{1}.caffemodel.h5'.format(snapshot_prefix, snapshot_iter)
            active_samples = samples.choose_active(
                    deploy_net_path,
                    caffemodel_path, mean_file, train_db_path,
                    batch_size, criterium, BATCHES_PER_RUN, input_shape, used_samples
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

































