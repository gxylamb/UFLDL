import numpy as np

def load_MNIST_images(filename):
	#return the 28*28*(num_of_images) matrix

	with open(filename,"r") as f:
		nothing=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		num_of_images=np.fromfile(f,dtype=np.dtype('>i4'),count=1)
		num_of_rows=np.fromfile(f,dtype=np.dtype('>i4'),count=1)
		num_of_cols=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		images=np.fromfile(f,dtype=np.ubyte)
		images=images.reshape((num_of_images,num_of_rows*num_of_cols)).transpose()
		images=images.astype(np.float64)/255

		f.close()

		return images


def load_MNIST_labels(filename):
	#return the (num_of_images)*1 contains the labels of MNIST images

	with open(filename,"r") as f:
		nothing=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		num_labels=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		labels=np.fromfile(f,dtype=np.ubyte)

		f.close()

		return labels



