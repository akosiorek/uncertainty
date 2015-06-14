import numpy as np
import matplotlib.pyplot as plt
import caffe
from scipy.stats import entropy
from operator import itemgetter
import lmdb # https://github.com/dw/py-lmdb/
import re

caffe_root = '../../../'

MODEL_FILE = '../uncertainty_deploy.prototxt'
PRETRAINED = '../snapshot/unc2_lambda_0_5_drop_0_5_htan.caffemodel'
LMDB_TEST =  caffe_root + 'examples/mnist/mnist_test_lmdb'

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(28, 28))

def sort_tuple_list_by_item( tuple_list, idx_item, descend=False ):

    return sorted( tuple_list, key=itemgetter(idx_item), reverse=descend )

def perform_tests():

	env = lmdb.open(LMDB_TEST, readonly=True)
	unc_correct_list   = []
	unc_incorrect_list = []
	with env.begin() as txn:
    		cursor = txn.cursor()

		index = 0

    		for key, raw_datum in cursor:
        		
			datum = caffe.proto.caffe_pb2.Datum()
			datum.ParseFromString(raw_datum)

			flat_x = np.fromstring(datum.data, dtype=np.uint8)
			x = flat_x.reshape(datum.channels, datum.height, datum.width)
			y = datum.label
			
			net.forward(data=x) 
        		probs =           net.blobs["probs"].data[0]
			uncertainty =     net.blobs["uncertainty"].data[0]
			uncertainty_raw = net.blobs["uncertainty_raw"].data[0]
			ip2 =             net.blobs["ip2"].data[0]
			ip_unc1 =            net.blobs["ip_unc1"].data[0]
			
			y_predicted = probs.argmax()

#			if uncertainty != 0:			
#			print "x:          ", x
			if y == y_predicted:
				unc_correct_list.append( uncertainty[0] )
			else:
				unc_incorrect_list.append( uncertainty[0] )
			#	print uncertainty

			# print "test image:     ", index			
			# print "y:              ", y
	                # print "y predicted:    ", y_predicted
#			print "uncertainty_ipu:", ip_unc1
			# print "probs_raw:      ", ip2
			# print "uncertainty_raw:", "%f" % uncertainty_raw
			# print "probs:          ", np.array(probs, dtype=np.dtype(float))
			# print "uncertainty:    ", "%f" % uncertainty
				
			# print "\n---\n"
			index += 1


	#print unc_correct_list
	#print unc_incorrect_list
	print "unc mean correct:   %.2f" % np.mean( unc_correct_list )
        print "unc std correct:    %.2f" % np.std( unc_correct_list )
	print "unc mean incorrect: %.2f" % np.mean( unc_incorrect_list )
	print "unc std incorrect:  %.2f" % np.std( unc_incorrect_list )
			#prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
			
	# print '%s\n' % image_filename

	#if prediction[0].shape[0] != 1000:

	#	print "Error: Prediction does not contain all 1000 classes."
	#	return

	#top3 = sort_tuple_list_by_item( zip(class_names, prediction[0]), 1, descend=True )[:3]

	#for i in xrange(3):
	#	print 'predicted class %i: %s' % ( i+1, top3[i][0] )
	#	print 'probability: %f' % top3[i][1]

	#print '\nentropy: %f' % entropy(prediction[0])

	#print '\n'
perform_tests()
