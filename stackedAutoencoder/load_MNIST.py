import numpy as np

def load_images(filename):

	with open(filename,'r') as f:

		magic=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		num_of_images=np.fromfile(f,dtype=np.dtype('>i4'),count=1)
		num_of_rows=np.fromfile(f,dtype=np.dtype('>i4'),count=1)
		num_of_cols=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		images=np.fromfile(f,dtype=np.ubyte)
		# 784*60000 
		images=images.reshape((num_of_images,num_of_cols*num_of_rows)).transpose()
		images=images.astype(np.float64)/255

		f.close()

		return images

def load_labels(filename):

	with open(filename,'r') as f:

		magic=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		num_of_labels=np.fromfile(f,dtype=np.dtype('>i4'),count=1)

		labels=np.fromfile(f,dtype=np.ubyte)

		f.close()

		return labels





