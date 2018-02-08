from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

import caffe

from utils import *

def weight_printing(input_net):
	for layer_name, param in input_net.params.iteritems():
		print type(layer_name)
		layer_weights_file = "models/bvlc_reference_caffenet/tmp/{}_weights.txt".format(layer_name)
		layer_weights_file2 = "models/bvlc_reference_caffenet/tmp/{}_weights2.txt".format(layer_name)
		layer_biases_file = "models/bvlc_reference_caffenet/tmp/{}_biases.txt".format(layer_name)
		np.savetxt(layer_weights_file, param[0].data.flat, delimiter=',',fmt="%f")
		np.savetxt(layer_weights_file2, input_net.params[layer_name][0].data.flat, delimiter=',',fmt="%f")
		np.savetxt(layer_biases_file, param[1].data.flat, delimiter=',',fmt="%f")
	print "weight_printing"

def main(model, input):
	caffe.set_mode_gpu()
	print "main"
	input_net = load_model(model, input)    
	weight_printing(input_net)

def parse_args():
	"""Parse input arguments
	"""

	parser = ArgumentParser(description=__doc__,
							formatter_class=ArgumentDefaultsHelpFormatter)

	parser.add_argument('model',
						help='model definition')
	parser.add_argument('input_weights',
						help='input weights')
	# parser.add_argument('output_weights',
						# help='output weights')
	# parser.add_argument('sparsity',
						# help='Sparsity, number of zeros')

	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = parse_args()

	main(args.model, args.input_weights)