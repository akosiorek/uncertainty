import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
	with open(path) as f:
		data = f.readlines()
	
	return np.asarray([line.strip() for line in data], dtype=np.float32)

def main():
	uncert = load_data('2_max_uncert.txt')
	labels = load_data('label.txt')

	positive = [];
	negative = [];

	for (label, u) in zip(labels, uncert):
		if label > 0.01:
			positive.append(u)
		elif 1:
			negative.append(u)

	print 'positive'
	print np.histogram(positive) 
	print 'negative'
	print np.histogram(negative)

	plt.hist(positive)
	plt.title("Correctly classified")
	plt.xlabel('Confidence')
	plt.ylabel("Num samples")
	plt.show()
	


if __name__ == '__main__':
	main()



