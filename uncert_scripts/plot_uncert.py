#!/usr/bin/python
"""
Creates 10-bin histograms separately for correctly
and incorrectly classified samples from text files
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    with open(path) as f:
        data = f.readlines()

    return np.asarray([line.strip() for line in data], dtype=np.float32)


def main(uncert_path, labels_path, ylim_pos=1, ylim_neg=1):
    uncert = load_data(uncert_path)
    labels = load_data(labels_path)

    name = os.path.splitext(uncert_path)[0];
    positive = [];
    negative = [];

    for (label, u) in zip(labels, uncert):
        if label > 0.01:
            positive.append(u)
        elif 1:
            negative.append(u)

    # print 'positive'
    # print np.histogram(positive)
    # print 'negative'
    # print np.histogram(negative)

    # plt.ylim(ymax=4500)

    def hist(x, ylim):
        bins = np.asarray(range(0, 11)) / 10.0
        weights = np.ones_like(x) / len(x)
        print bins
        data, bins = np.histogram(x, weights=weights, bins=bins)
        print data
        print bins
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        plt.ylim((0, float(ylim)))
        plt.xlim((0, 1))
        plt.bar(center, data, align='center', width=width)

    hist(positive, ylim_pos)
    plt.title("Correctly classified")
    plt.xlabel('Uncertainty')
    plt.ylabel("Num samples")
    plt.savefig("%s_positive.png" % (name,))
    plt.close()

    hist(negative, ylim_neg)
    plt.title("Falsly classified")
    plt.xlabel('Uncertainty')
    plt.ylabel("Num samples")
    plt.savefig("%s_negative.png" % (name,))
    plt.close()


if __name__ == '__main__':
    main(*sys.argv[1:])
