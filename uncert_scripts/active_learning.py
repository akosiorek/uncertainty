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
import os
from subprocess import call
from evaluate_net import active_samples


EXEC = 'build/tools/caffe'
MODEL_FOLDER = 'models/uncertainty'
SOLVER_INIT = 'solver.prototxt'  # solver used to initialize the network
SOLVER = 'solver_al.prototxt'   # solver used for active learning
DEPLOY_NET = 'deploy.prototxt'


def train_cmd(solver):
    solver_file = os.path.join(MODEL_FOLDER, solver)
    return '{0} train --solver={1}'.format(EXEC, solver_file)

# 1. Initialize the net by training with an initial solver
call(train_cmd(SOLVER_INIT).split(' '))


# 2. Evaluate uncertaintities and put appropriate samples/labels in a new db
    samples_to_extract = active_samples(DEPLOY_NET)




