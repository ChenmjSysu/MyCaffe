import numpy as np
import sys
import random
import os
import optparse
from os.path import isfile, join
import caffe
from os import listdir
import cPickle
from scipy.signal import convolve2d
import matplotlib.pyplot as ply
from PIL import Image
import PIL
import operator
import matplotlib.cm as cm
from scipy.misc import imresize
% matplotlib inline
from scipy.stats import pearsonr, spearmanr
from scipy.io import savemat
from sklearn import metrics

caffe_root = "/home/chen/code/caffe-master/"
sys.path.insert(1, caffe_root + "python")

cross_number = 5
train_portion = 0.8
test_portion = 0.2
root = ""
save_root = ""
background_root = ""
foreground_root = ""

imagenet_model = ""
woratiofine_folder_name = ""
imagenet_deploy = ""
urban_deploy = ""
classes = []

cross_portion = 0.0
urban_model = ""
net = caffe.Classifier(urban_deploy, urban_model, image_dims=[256, 256])
net.set_phase_test()
net.set_mode_cpu()
net.set_mean("data", "")
net.set_channel_swap("data", (2, 1, 0))
net.set_input_scale("data", 255)

def get_pool_indice(net, layer_number, no_feature_map):
	pool_layer = "pool" + str(layer_number)
	conv_layer = "conv" + str(layer_number)
	pool_indice = np.zeros(net.blobs[pool_layer].data.shape, dtype="int")
	#print pool_indice.shape
	for i in xrange(0, net.blobs[conv_layer].data.shape[2] - 1, 2):
		for j in xrange(0, net.blobs[conv_layer].data.shape[3] - 1, 2):
			temp = net.blobs[conv_layer].data[4, :, i : i + 3, j : j + 3]
			# print temp.shape
			if temp.shape[1] != 3:
				temp = np.append