
import os
os.environ['GLOG_minloglevel'] = '1'

import numpy as np
np.seterr(invalid='raise')

OUTPUT_LAYERS = ['ip2']
CAFFE_EXEC = '../build/tools/caffe'
MODEL_FOLDER = 'models/uncertainty'
POSTFIX = '.active'
EPOCH_FILE = 'epochs.txt'

MAX_EPOCHS = 10
BATCHES_PER_RUN = 100
ITERS_TO_INIT = 1000
MAX_ITER = 5000
SNAPSHOT_EVERY_ITER = 1000
DROPOUT_ITERS = 10
UNCERTAINTY_THRESOLD = 0
LOG_EVERY = 10000
NEW_SAMPLES_PER_ITER = 2500