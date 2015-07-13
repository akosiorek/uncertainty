#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
	with open(path) as f:
		data = f.readlines()
	
	return np.asarray([line.strip() for line in data], dtype=np.float32)

def main(uncert_path, labels_path):
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

	#print 'positive'
	#print np.histogram(positive) 
	#print 'negative'
	#print np.histogram(negative)

	plt.hist(positive)
	plt.title("Correctly classified")
	plt.xlabel('Confidence')
	plt.ylabel("Num samples")
	plt.savefig("%s_positive.png" % (name, ))
	plt.close()

	plt.hist(negative)
	plt.title("Falsly classified")
	plt.xlabel('Confidence')
	plt.ylabel("Num samples")
	plt.savefig("%s_negative.png" % (name, ))
	plt.close()

if __name__ == '__main__':
	main(*sys.argv[1:])



