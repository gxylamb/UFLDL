import load_MNIST
import sparse_autoencoder
import softmax
import scipy.optimize
import numpy as np

input_size=28*28
num_of_classes=10
hidden_size_L1=200
hidden_size_L2=150
sparsity_param=0.1
_lambda=3e-3
beta=3

train_images=load_MNIST.load_images('train-images.idx3-ubyte')
train_labels=load_MNIST.load_labels('train-labels.idx1-ubyte')

#==========================================================================================================================
sae1_theta=sparse_autoencoder.initialize(hidden_size_L1,input_size)
J=lambda x:sparse_autoencoder.sparse_autoencoder_cost(x,input_size,hidden_size_L1,_lambda,sparsity_param,beta,train_images)
_options={'maxiter':400,'disp':True}

result=scipy.optimize.minimize(J,sae1_theta,method='L-BFGS-B',jac=True,options=_options)
sae1_opt_theta=result.x

print(result)
#==========================================================================================================================
#==========================================================================================================================
sae1_feature=sparse_autoencoder.sparse_autoencoder(sae1_opt_theta,hidden_size_L1,input_size,train_images)

sae2_theta=sparse_autoencoder.initialize(hidden_size_L2,hidden_size_L1)
J=lambda x:sparse_autoencoder.sparse_autoencoder_cost(x,hidden_size_L1,hidden_size_L2,_lambda,sparsity_param,beta,sae1_feature)
_options={'maxiter':400,'disp':True}

result=scipy.optimize.minimize(J,sae2_theta,method='L-BFGS-B',jac=True,options=_options)

sae2_opt_theta=result.x

print(result)
#==========================================================================================================================

sae2_feature=sparse_autoencoder.sparse_autoencoder(sae2_opt_theta,hidden_size_L2,hidden_size_L1,sae1_feature)
_options={'maxiter':400,'disp':True}

softmax_theta,softmax_input_size,softmax_num_classes=softmax.softmax_train(hidden_size_L2,num_of_classes,_lambda,sae2_feature,train_labels,_options)

print(softmax_theta)