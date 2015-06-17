#!/usr/bin/env python

import sys
import os.path
import re

UNCERT_EXT = 'uncert.prototxt'
DEPLOY_EXT = 'deploy.prototxt'


DEPLOY_NAME = 'UncertaintyDeployNet'
DEPLOY_HEADER = '\n'.join(['input: "data"', 'input_dim: 1', 'input_dim: 1', 'input_dim: 28', 'input_dim: 28'])
DEPLOY_BANNED_LAYERS = ['Data', 'Loss', 'Accuracy']

deploy_layers_expr = re.compile('|'.join(DEPLOY_BANNED_LAYERS), re.I)
def create_deploy_model(layers):
	layer_filter = lambda x: not deploy_layers_expr.search(x)
	deploy_layers = [layer for layer in layers if layer_filter(layer)]
	
	layer = 'name: %s\n\n%s\n\n%s' %(DEPLOY_NAME, DEPLOY_HEADER, '\n'.join(deploy_layers))
	return layer



zero_learning_rate_expr = re.compile(r'(s*param\s*{\s*lr_mult:\s*)([1-9])(\s*})')
nonzero_learning_rate_expr = re.compile(r'(s*param\s*{\s*lr_mult:\s*)(0)(\s*})')
def modify_learning_rates(layer, weight_lr, bias_lr):
	if weight_lr != 0:
		learning_rate_expr = nonzero_learning_rate_expr
	else:
		learning_rate_expr = zero_learning_rate_expr

	if learning_rate_expr.search(layer):
		layer = learning_rate_expr.sub(r'\1 ' + str(weight_lr) + r'\3 ', layer, 1)
		layer = learning_rate_expr.sub(r'\1 ' + str(bias_lr) + r'\3 ', layer, 1)

	return layer
	

def create_uncert_model(layers):
	uncertainty_filter = lambda x: 'unc' in x
	uncert_layers = []
	for layer in layers:
		if uncertainty_filter(layer):
			layer = modify_learning_rates(layer, 1, 2)
		else: # classification layer or no lr
			layer = modify_learning_rates(layer, 0, 0)
		uncert_layers.append(layer)
	
	return '\n'.join(uncert_layers)

def write_file(filename, data):
	with open(filename, 'w') as f:
		f.write(data)

def main(args):
	basefile = args[0]
	if not os.path.isfile(basefile):
		print 'Couldn\' open the file: ', basefile
		sys.exit(1)
	
	ext = os.path.splitext(basefile)[1]
	if ext != '.prototxt':
		print 'File is not a *.prototxt file'
		sys.exit(1)
	

	name = '_'.join(basefile.split('_')[:-1])
	uncert_modelfile = '_'.join([name, UNCERT_EXT])
	deploy_modelfile = '_'.join([name, DEPLOY_EXT])
	
	layers = open(basefile).readlines()
	#layers = [layer for layer in layers if layer.strip()]
	name = layers[0]
	layers = [layer for layer in layers if layer]
	layers = ''.join(layers[1:]).split('layer')
	layers = [' layer' + layer for layer in layers if layer.strip()]
	

	write_file(uncert_modelfile, create_uncert_model(layers))
	write_file(deploy_modelfile, create_deploy_model(layers))

	print 'created models:'
	print 'deploy:\t', deploy_modelfile
	print 'uncert:\t', uncert_modelfile

if __name__ == '__main__':
	main(sys.argv[1:])
