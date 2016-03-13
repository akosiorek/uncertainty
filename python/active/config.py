
OUTPUT_LAYERS = ['ip2']
CAFFE_EXEC = '../build/tools/caffe'
MODEL_FOLDER = 'models/uncertainty'
POSTFIX = '.active'
EPOCH_FILE = 'epochs.txt'

MAX_EPOCHS = 10
BATCHES_PER_RUN = 100
ITERS_TO_INIT = 1000
DROPOUT_ITERS = 10