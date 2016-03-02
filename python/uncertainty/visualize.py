#!/usr/bin/env python
"""
Plots accuracy and mean uncertinty plus that of correctly and incorrectly classified samples.
Assumes appropriate folder structure and correctnes and labels in separate files
"""
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np


def value(x):
    return int(x.split('_')[1].split('.')[0])


def filter_and_sort(header, files):
    l = [f for f in files if f.startswith(header)]
    return sorted(l, key=value)


def read_file(path):
    with open(path) as f:
        samples = f.read().split('\n')

    return np.asarray([float(s.strip()) for s in samples if s])


if __name__ == '__main__':
    folder = sys.argv[1]
    files = os.listdir(folder)

    label_files = filter_and_sort('label', files)
    uncert_files = filter_and_sort('uncert', files)

    max_file = uncert_files[-1]
    max_num = os.path.splitext(max_file)[0].split('_')[-1]
    max_num = int(max_num)

    mean_uncert = []
    mean_uncert_pos = []
    mean_uncert_neg = []
    mean_acc = []
    uncertss = []

    x = []
    for (label_file, uncert_file) in zip(label_files, uncert_files):
        x.append(int(re.search(r'_(\d+)', label_file).groups()[0]))
        labels = read_file(os.path.join(folder, label_file))
        uncerts = read_file(os.path.join(folder, uncert_file))

        uncertss.append(uncerts)

        mean_uncert.append(uncerts.mean())
        mean_uncert_pos.append(uncerts[labels > 0].mean())
        mean_uncert_neg.append(uncerts[labels == 0].mean())
        mean_acc.append(labels.mean())

    plt.figure()
    plt.plot(x, mean_acc, 'g', label='accuracy', linewidth=2)
    plt.plot(x, mean_uncert, 'r', label='uncertainty', linewidth=2)
    plt.plot(x, mean_uncert_pos, 'y', label='uncertainty pos', linewidth=2)
    plt.plot(x, mean_uncert_neg, 'm', label='uncertainty neg', linewidth=2)
    plt.title('Uncertainty and Accuracy for CIFAR-10')
    plt.legend(loc='best')
    plt.xlabel('Number of training iterations')
    plt.grid()
    plt.savefig('uncert_and_acc_plot.png')

    uncertss = zip(*uncertss)
    plt.figure()
    for i in xrange(50, 55):
        plt.plot(x, uncertss[i], linewidth=2, label=str(np.mean(uncertss[i])))

    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('Uncertainty')
    plt.xlabel('Number of training iterations')
    plt.savefig('sample_uncert_evolution.png')

    scores = zip(x, mean_uncert)
    scores = sorted(scores, key=lambda x:x[0])
    for s in scores:
        print 'iter = {:06}, uncertainty = {}'.format(*s)
