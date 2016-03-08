import os
import sys
import re
import shutil

import numpy as np
from sklearn.neighbors import NearestNeighbors


def wait_bar(pre_msg, post_msg, counter, limit):
    print '{0} {1}/{2} {3}\r'.format(pre_msg, counter, limit, post_msg),
    sys.stdout.flush()


def entropy(x):
    y = x / np.tile(np.sum(x, axis=1)[:, np.newaxis], (1, x.shape[1]))
    # print '1', np.where(y == 1), '0', np.where(y == 0), y.shape

    index = (y != 0)
    y[index] = y[index] * np.log(y[index])
    e = -np.sum(y, axis=1) / np.log(y.shape[-1])
    return e


def softmax(x):
    e = np.exp(x)
    return e / np.tile(np.sum(e, axis=1)[:, np.newaxis], (1, x.shape[1]))


def second_max(x):
    argmax = x.argmax(axis=1)
    linear_index = range(x.shape[0])
    m1 = x[linear_index, argmax]
    y = x.copy()
    y[linear_index, argmax] = 0
    m2 = y.max(axis=1)
    return m2 / m1


def write_to_file(path, container):
    with open(path, 'w') as f:
        f.write('\n'.join([str(e) for e in container]))


def get_snapshot_number(path):
    return int(re.search(r'_(\d+)', path).groups()[0])


def get_snapshot_files(snapshot_folder, solverstate=False, every_iter=None):

    if solverstate:
        model_type = 'solverstate'
    else:
        model_type = 'caffemodel'

    files = os.listdir(snapshot_folder)
    files = [[get_snapshot_number(f), f] for f in files if model_type in f]
    files = sorted(files, key=lambda f: f[0])

    if every_iter is not None:
        indices = [0]
        last = files[0][0] + every_iter

        nn = NearestNeighbors()
        nums = np.asarray([f[0] for f in files])[:, np.newaxis]
        nn.fit(nums)
        while last <= files[-1][0]:
            ind = nn.kneighbors(last, 1, return_distance=False)[0]
            if ind != indices[-1]:
                indices.append(ind[0])
            last += every_iter

        if indices[-1] != len(files)-1:
            indices.append(len(files)-1)

        files = [files[i] for i in indices]

    files = [(f[0], os.path.join(snapshot_folder, f[1])) for f in files]
    return files


def write_to_file(path, container, sep='\n'):
    with open(path, 'w') as f:
        f.write(sep.join([str(e) for e in container]))


def create_temp_path(path):
    return os.path.join('/tmp', os.path.basename(path))


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def append_to_file(path, msg, sep='\n'):
    with open(path, 'a') as f:
        f.write(msg)
        f.write(sep)

if __name__ == '__main__':
    for num, snapshot in get_snapshot_files(sys.argv[1], every_iter=int(sys.argv[2])):
        print num, snapshot