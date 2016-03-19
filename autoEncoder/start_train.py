import numpy as np
#import scipy.optimize
import sparse_autoencoder
#import gradient
import load_MNIST

#==========================================================
#Step 0 initialize the parameters
visible_size=28*28
hidden_size=196

#which is rho in the tutorial
sparsity_param=0.1
#which is the weight decay parameter in J(W,b;x,y)
_lambda=3e-3
#which is the sparsity penalty term
beta=3
#==========================================================

#==========================================================
#Step 1 Load images
#
#load 10000 images from MNIST
#remember to modify the route
images=load_MNIST.load_MNIST_images('data\\mnist\\train-images-idx3-ubyte')
patches=images[:,0:10000]

theta=sparse_autoencoder.initialize(hidden_size,visible_size)
(cost,grad)=sparse_autoencoder.sparse_autoencoder_cost(theta,visible_size,hidden_size,_lambda,sparsity_param,beta,patches)

print (cost,grad)
#==========================================================

#==========================================================
#Step 2 Gradient Check
#maybe later..
#==========================================================

#==========================================================
#Step 3 Visualization
#PIL not for Python3 so also maybe later
#==========================================================


