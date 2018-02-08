from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

import caffe

from utils import *

def load_stats(input_net):
	for layer_name, param in input_net.params.iteritems():
		print type(layer_name)
		if len(param[0].data.shape) == 4:
			o_shape = param[0].data.shape[0]
			i_shape = param[0].data.shape[1]
			h = param[0].data.shape[2]
			w = param[0].data.shape[3]
			counts = o_shape * i_shape * h * w
			zero_counts = counts - np.count_nonzero(param[0].data)
			sparsity = zero_counts / float(counts)
			layer_power_load_file = "models/bvlc_reference_caffenet/tmp/powering_self_final_{}_load.txt".format(layer_name)
			print layer_power_load_file
			f = open(layer_power_load_file, 'wb')
			load = []
			for o_idx in xrange(o_shape):
				# cur_load = 0
				f.write("o_idx: {}\n".format(str(o_idx)))
				for i_idx in xrange(i_shape):
					f.write("i_idx: {}\n".format(str(i_idx)))
					f.write("cur_kernel: {}\n".format(str(param[0].data[o_idx][i_idx])))
					f.write("cur_load: {}\n".format(str(np.count_nonzero(param[0].data[o_idx][i_idx]))))
					load.append(np.count_nonzero(param[0].data[o_idx][i_idx]))
			f.write("max_load: {}\n".format(str(np.max(load))))
			f.write("min_load: {}\n".format(str(np.min(load))))
			f.write("sparsity: {}\n".format(str(sparsity)))
			f.close()

def weight_stats(input_net):
	for layer_name, param in input_net.params.iteritems():
		print type(layer_name)
		if len(param[0].data.shape) == 4:
			cur_layer_params_o = param[0].data.shape[0]
			print cur_layer_params_o
			cur_layer_params_i = param[0].data.shape[1]
			print cur_layer_params_i
			cur_layer_params_h = param[0].data.shape[2]
			print cur_layer_params_h
			cur_layer_params_w = param[0].data.shape[3]
			print cur_layer_params_w
			cur_layer_params = cur_layer_params_o * cur_layer_params_i * cur_layer_params_h * cur_layer_params_w
		elif len(param[0].data.shape) == 2:
			cur_layer_params_o = param[0].data.shape[0]
			print cur_layer_params_o
			cur_layer_params_i = param[0].data.shape[1]
			print cur_layer_params_i
			cur_layer_params = cur_layer_params_o * cur_layer_params_i
		filter = param[0].data.flat
		unique, counts = np.unique(filter, return_counts=True)
		# print type(unique)
		# print type(counts)
		# print np.asarray((unique, counts)).T
		np.savetxt('models/bvlc_reference_caffenet/tmp/powering_self_final_stat_' + layer_name + '_count.txt', np.asarray((unique, counts)).T, delimiter=',',fmt="%s")
		np.savetxt('models/bvlc_reference_caffenet/tmp/powering_self_final_stat_' + layer_name + '_stat.txt', np.asarray((unique, counts/float(cur_layer_params))).T, delimiter=',',fmt="%s")

# def sparsity_ratio():

def main(model, input):
	caffe.set_mode_gpu()
	print "main"
	input_net = load_model(model, input)
	weight_stats(input_net)
	# load_stats(input_net)

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