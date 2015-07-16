import numpy as np
import matplotlib.pyplot as plt
import caffe
from scipy.stats import entropy
from operator import itemgetter
import lmdb # https://github.com/dw/py-lmdb/
import re

caffe_root = '../../../'

MODE = 'train'
LMDB_IP1 =  MODE + '_lmdb_lenet_ip1'
LMDB_IP2 =  MODE + '_lmdb_lenet_ip2'
LMDB_LABEL = MODE + '_lmdb_lenet_label'


def sort_tuple_list_by_item( tuple_list, idx_item, descend=False ):

    return sorted( tuple_list, key=itemgetter(idx_item), reverse=descend )


def get_data(db):
	data = []
	with db as txn:
		cursor = txn.cursor()
		for key, row_datum in cursor:
			datum = caffe.proto.caffe_pb2.Datum()
			datum.ParseFromString(row_datum)
			data.append(np.fromstring(datum.data, dtype=np.uint8)
	return data

def balanced_num(label_db, probs_db):
	labels = get_data(label_db)
	probs = get_data(probs_db)

	predicted_class = np.argmax(probs, 1);
	correct = predicted_class == labels
	num_correct = np.sum(correct)
	num_incorrect = labels.shape[0] - num_correct;
	return (num_correct, num_incorrect);

def main():

	ip1_db = lmdb.open(LMDB_IP1, readonly=False)
	ip2_db = lmdb.open(LMDB_IP2, readonly=False)
	label_db = lmdb.open(LMDB_LABEL, readonly=False)

	num_correct, num_incorrect = balanced_num(label_db, ip2_db)

	multiply_incorrect = num_correct / num_incorrect
	if multiply_incorrect == 0:
		print 'Data is already balanced'
		return

	

	



	txn_ip1 = data_ip1.begin();
	txn_ip2 = data_ip2.begin();
	txn_label = data_label.begin();
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
			ip1 =             net.blobs["ip1"].data[0]
			ip2 =             net.blobs["ip2"].data[0]
			ip_unc1 =         net.blobs["ip_unc1"].data[0]
			
			y_predicted = probs.argmax()

#			if uncertainty != 0:			
#			print "x:          ", x
			if y == y_predicted:
				unc_correct_list.append( uncertainty[0] )
			else:
				unc_incorrect_list.append( uncertainty[0] )
			#	print uncertainty
			
			print "test image:     ", index			
			print "y:              ", y
	                print "y predicted:    ", y_predicted
#			print "uncertainty_ipu:", ip_unc1
#			print "ip1:            ", ip1
			print "probs_raw:      ", ip2
			print "uncertainty_raw:", "%f" % uncertainty_raw
			print "probs:          ", np.array(probs, dtype=np.dtype(float))
			# print "uncertainty:    ", "%f" % uncertainty
				
			print "\n---\n"
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
