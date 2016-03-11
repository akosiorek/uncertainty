#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
from visualize import compute_accuracy_uncertainty

if __name__ == '__main__':
    output = sys.argv[1]
    folders = sys.argv[2:]

    title = os.path.basename(output)
    title = os.path.splitext(title)[0]
    title = ' '.join([word.capitalize() for word in title.split('_')])

    results = [compute_accuracy_uncertainty(folder) for folder in folders]
    plt.figure()
    for (num, result), folder in zip(enumerate(results), folders):

        name = os.path.basename(folder)
        if len(name) == 0:
            name = folder

        color = colors.cnames.values()[num]

        x = result['iter']
        plt.plot(x, result['accuracy'], color=color, linestyle='-', label='{0} acc'.format(name), linewidth=2)
        plt.plot(x, result['mean_uncert'], color=color, linestyle='-.', label='{0} unc'.format(name), linewidth=2)

    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('Number of training iterations')
    plt.grid()
    plt.savefig(output)