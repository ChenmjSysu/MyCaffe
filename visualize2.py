# display each layer's blob
for layer_name, blob in net.blobs.iteritems():
	print layer_name + "\t" + str(blob.data.shape)

# display each layer's param
for layer_name, param in net.param.iteritems():
	print layer_name + "\t" + str(param[0].data.shape), str(param[1].data.shape)



def vis_square(data):
	# normalize
	data = (data - data.min()) / (data.max() - data.min())

	n = int(np.ceil(np.sqrt(data.shape[0])))
	# add padding between patch
	paddding = (((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0), ) * (data.ndim - 3))
	# tile the filter into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

	data = data.reshoae((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

	# display the images
	plt.imshow(data);plt.axis("off")


# display the filters of conv1
filters = net.params["conv1"][0].data
vis_square(filters.transpose(0, 2, 3, 1))

# display the feature map of conv1
feat = net.blobs["conv1"].data[0, : 36]
vis_square(feat)