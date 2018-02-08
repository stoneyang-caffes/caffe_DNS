from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import caffe

from utils import *

def weights_masking(dns_param, dns_params, normal_param, normal_params, logs_root):
	for dns_pr, n_pr in zip(dns_param, normal_param):
		print dns_pr, n_pr
		W = np.asarray(dns_params[dns_pr][0])
		T = np.asarray(dns_params[dns_pr][2])
		W_masked = np.multiply(W, T)
		print W_masked.shape
		normal_params[n_pr][0].flat = W_masked.flat
		# print normal_params[n_pr][0].shape
		W_n_file = logs_root + '{}_Wn.txt'.format(n_pr)
		b = np.asarray(dns_params[dns_pr][1])
		bT = np.asarray(dns_params[dns_pr][3])
		b_masked = np.multiply(b, bT)
		normal_params[n_pr][1][...] = b_masked
		b_masked_file = logs_root + '{}_bn.txt'.format(n_pr)
		np.savetxt(W_n_file, normal_params[n_pr][0].flat, delimiter=',',fmt="%f")
		np.savetxt(b_masked_file, normal_params[n_pr][1][...], delimiter=',',fmt="%f")

def weights_printing(net, param, logs_root):
	params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in param}
	for pr in params:
		W_n_shape = params[pr][0].shape
		b_n_shape = params[pr][1].shape
		print '{} weights are {} dimensional and biases are {} dimensional'.format(pr, W_n_shape, b_n_shape)
	return params

def weights_masks_printing(net, param, logs_root):
	params = {pr: (net.params[pr][0].data, net.params[pr][1].data, net.params[pr][2].data, net.params[pr][3].data) for pr in param}
	for pr in params:
		print pr
		W_shape = params[pr][0].shape
		b_shape = params[pr][1].shape
		T_shape = params[pr][2].shape
		bT_shape = params[pr][3].shape
		print '{} weights are {} dimensional and biases are {} dimensional'.format(pr, W_shape, b_shape)
		print '{} weight masks are {} dimensional and bias masks are {} dimensional'.format(pr, T_shape, bT_shape)
		# save'em
		print 'saving {} to txt ....'.format(pr)
		W_file  = logs_root + '{}_W.txt'.format(pr)
		T_file  = logs_root + '{}_T.txt'.format(pr)
		b_file  = logs_root + '{}_b.txt'.format(pr)
		bT_file = logs_root + '{}_bT.txt'.format(pr)
		np.savetxt(W_file, params[pr][0].flat, delimiter=',',fmt="%f")
		np.savetxt(T_file, params[pr][2].flat, delimiter=',',fmt="%f")
		np.savetxt(b_file, params[pr][1][...], delimiter=',',fmt="%f")
		np.savetxt(bT_file, params[pr][3][...], delimiter=',',fmt="%f")
		# dry masking, for checking
		W = np.asarray(params[pr][0])
		T = np.asarray(params[pr][2])
		W_masked = np.multiply(W, T)
		b = np.asarray(params[pr][1])
		bT = np.asarray(params[pr][3])
		b_masked = np.multiply(b, bT)
		W_masked_file = logs_root + '{}_W_masked.txt'.format(pr)
		b_masked_file = logs_root + '{}_b_masked.txt'.format(pr)
		np.savetxt(W_masked_file, W_masked.flat, delimiter=',',fmt="%f")
		np.savetxt(b_masked_file, b_masked.flat, delimiter=',',fmt="%f")
		print 'saving {} to txt done! '.format(pr)
	return params

def main(dns_model, dns_weights, normal_model, normal_weights, logs_root):
	# load DNS model
	dns_net = load_model(dns_model, dns_weights)
	# parse DNS model
	dns_param = parse_mask_model(dns_net)
	print dns_param
	# save W & T for weights & biases to txt
	dns_params = weights_masks_printing(dns_net, dns_param, logs_root)
	# load normal model
	normal_net = load_model_wo_weights(normal_model)
	# parse normal model
	normal_param = parse_model(normal_net)
	normal_params = weights_printing(normal_net, normal_param, logs_root)
	# do the masking for weights & biases, and save'em to txt
	weights_masking(dns_param, dns_params, normal_param, normal_params, logs_root)
	# save normal model
	normal_net.save(normal_weights)

def parse_args():
	"""Parse input arguments
	"""

	parser = ArgumentParser(description=__doc__,
							formatter_class=ArgumentDefaultsHelpFormatter)

	parser.add_argument('dns_model',
						help='dns model definition')
	parser.add_argument('dns_weights',
						help='dns weights')
	parser.add_argument('normal_model',
						help='normal model')
	parser.add_argument('normal_weights',
						help='normal weights')
	parser.add_argument('logs_root',
						help='logs root')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	# portal
	main(args.dns_model, args.dns_weights, args.normal_model, args.normal_weights, args.logs_root)