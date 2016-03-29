import struct
import array
import time

import scipy.sparse
import scipy.optimize
import numpy as np

def loadImages(filename):
	image_file=open(filename,'rb')

	#as we know one Hexadecimal number should be 4 bit
	#according to the format
	#we should read 4 byte(32bit) which is 8 Hexadecimal number
	magic=image_file.read(4)
	num_images_hex=image_file.read(4)
	num_rows_hex=image_file.read(4)
	num_cols_hex=image_file.read(4)

	num_images=struct.unpack('>I',num_images_hex)[0]
	num_rows=struct.unpack('>I',num_rows_hex)[0]
	num_cols=struct.unpack('>I',num_cols_hex)[0]

	#initialize the data
	data=np.zeros((num_rows*num_cols,num_images))

	all_images=array.array('B',image_file.read())
	for i in range(num_images):
		data[:,i]=all_images[num_rows*num_cols*i:num_rows*num_cols*(i+1)]

	#we have to normalize
	return data/255

def loadLabels(filename):
	label_file=open(filename,'rb')

	magic=label_file.read(4)
	num_labels_hex=label_file.read(4)

	num_labels=struct.unpack('>I',num_labels_hex)[0]

	"""
	so this should be 
	array([[0],
       [0],
       [0],
       ..., 
       [0],
       [0],
       [0]])

	"""
	labels=np.zeros((num_labels,1),dtype=np.int)

	all_labels=array.array('b',label_file.read())
	label_file.close()

	labels[:,0]=all_labels[:]

	return labels

class softmaxReg(object):


	def __init__(self,input_size,num_classes,_lambda):
		self.input_size=input_size
		self.num_classes=num_classes
		self._lambda=_lambda

		rand=np.random.RandomState(int(time.time()))

		#so we got K thetas
		#the shape should be (28*28*10) * 1
		self.theta=0.005*np.asarray(rand.normal(size=(num_classes*input_size,1)))

	def getGroundTruth(self,labels):

		labels=np.array(labels).flatten()
		data=np.ones(len(labels))
		#index from 1 to len(labels)
		indptr=np.arange(len(labels)+1)

		ground_truth=scipy.sparse.csr_matrix((data,labels,indptr))
		ground_truth=np.transpose(ground_truth.todense())

		return ground_truth

	def softmaxCost(self,theta,input,labels):
		ground_truth=self.getGroundTruth(labels)
		theta=theta.reshape(self.num_classes,self.input_size)

		theta_x=np.dot(theta,input)
		exp=np.exp(theta_x)
		prob=exp/np.sum(exp,axis=0)

		#calc the cost term
		#groud_truth should be 11*60000 ?
		#act like one-hot encode
		loss=np.multiply(ground_truth,np.log(prob))
		traditional_loss=-(np.sum(loss)/input.shape[1])

		#calc the weight decay
		theta_square=np.multiply(theta,theta)
		weight_decay=0.5*self._lambda*np.sum(theta_square)

		cost=traditional_loss+weight_decay

		theta_grad=-np.dot(ground_truth-prob,np.transpose(input))
		theta_grad=theta_grad/input.shape[1]+self._lambda*theta
		theta_grad=np.array(theta_grad)
		theta_grad=theta_grad.flatten()

		return [cost,theta_grad]
		
	def softmaxPredict(self,theta,input):

		theta=theta.reshape(self.num_classes,self.input_size)

		theta_x=np.dot(theta,input)
		exp=np.exp(theta_x)
		prob=exp/np.sum(exp,axis=0)

		predictions=np.zeros((input.shape[1],1))
		predictions[:,0]=np.argmax(prob,axis=0)

		return predictions




def executeSoftmax():

	input_size=28*28
	num_classes=10
	_lambda=0.001
	max_iteration=100

	training_data=loadImages('train-images.idx3-ubyte')
	training_labels=loadLabels('train-labels.idx1-ubyte')

	regressor=softmaxReg(input_size,num_classes,_lambda)

	opt_solution=scipy.optimize.minimize(regressor.softmaxCost,regressor.theta,args=(training_data,training_labels),method='L-BFGS-B',jac=True,options={'maxiter': 100,'disp': True})
	opt_theta=opt_solution.x

	test_data=loadImages('t10k-images.idx3-ubyte')
	test_labels=loadLabels('t10k-labels.idx1-ubyte')

	predictions=regressor.softmaxPredict(opt_theta,test_data)

	accuracy=(test_labels[:,0]==predictions[:,0])

	print("accuracy is ",np.mean(accuracy))


executeSoftmax()










