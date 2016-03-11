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
    files = [f for f in files if f.endswith('.txt')]
    l = [f for f in files if f.startswith(header)]
    return sorted(l, key=value)


def read_file(path):
    with open(path) as f:
        samples = f.read().split('\n')

    return np.asarray([float(s.strip()) for s in samples if s])


def compute_accuracy_uncertainty(results_folder):
    files = os.listdir(results_folder)

    label_files = filter_and_sort('label', files)
    uncert_files = filter_and_sort('uncert', files)

    result = {
        'mean_uncert': [],
        'mean_uncert_pos': [],
        'mean_uncert_neg': [],
        'uncertainties': [],
        'accuracy': [],
        'iter': []
    }

    for (label_file, uncert_file) in zip(label_files, uncert_files):
        labels = read_file(os.path.join(results_folder, label_file))
        uncerts = read_file(os.path.join(results_folder, uncert_file))

        result['mean_uncert'].append(uncerts.mean())
        result['mean_uncert_pos'].append(uncerts[labels > 0].mean())
        result['mean_uncert_neg'].append(uncerts[labels == 0].mean())
        result['uncertainties'].append(uncerts)
        result['accuracy'].append(labels.mean())
        result['iter'].append(int(re.search(r'_(\d+)', label_file).groups()[0]))

    return result


if __name__ == '__main__':
    folder = sys.argv[1]
    result = compute_accuracy_uncertainty(folder)


    plt.figure()
    x = result['iter']
    plt.plot(x, result['accuracy'], 'g', label='accuracy', linewidth=2)
    plt.plot(x, result['mean_uncert'], 'r', label='uncertainty', linewidth=2)
    plt.plot(x, result['mean_uncert_pos'], 'y', label='uncertainty pos', linewidth=2)
    plt.plot(x, result['mean_uncert_neg'], 'm', label='uncertainty neg', linewidth=2)
    plt.title('Uncertainty and Accuracy for CIFAR-10')
    plt.legend(loc='best')
    plt.xlabel('Number of training iterations')
    plt.grid()
    plt.savefig('uncert_and_acc_plot.png')

    uncertss = zip(*result['uncertainties'])
    plt.figure()
    for i in xrange(50, 55):
        plt.plot(x, uncertss[i], linewidth=2, label=str(np.mean(uncertss[i])))

    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('Uncertainty')
    plt.xlabel('Number of training iterations')
    plt.savefig('sample_uncert_evolution.png')

    scores = zip(x, result['mean_uncert'], result['accuracy'])
    scores = sorted(scores, key=lambda x:x[0])
    for s in scores:
        print 'iter = {:6}, uncertainty = {}, accuracy = {}'.format(*s)
