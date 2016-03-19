import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))

def KL_div(x,y):
	return x*np.log(x/y)+(1-x)*np.log((1-x)/(1-y))

def initialize(hidden_size,visible_size):
	#random base r for W1 & W2
	r=np.sqrt(6)/np.sqrt(hidden_size+visible_size+1)

	W1=np.random.random((hidden_size,visible_size))*2*r-r
	W2=np.random.random((visible_size,hidden_size))*2*r-r

	b1=np.zeros(hidden_size,dtype=np.float64)
	b2=np.zeros(visible_size,dtype=np.float64)

	theta=np.concatenate((W1.reshape(hidden_size*visible_size),
							W2.reshape(hidden_size*visible_size),
							b1.reshape(hidden_size),
							b2.reshape(visible_size)))
	return theta

#visible size should be 8*8=64
#hidden size should be 25
#data(:,i) should be the ith patch of the example

def sparse_autoencoder_cost(theta,visible_size,hidden_size,_lambda,sparsity_param,beta,data):
        W1=theta[0:hidden_size*visible_size].reshape(hidden_size,visible_size)
        W2=theta[hidden_size*visible_size:2*hidden_size*visible_size].reshape(visible_size,hidden_size)
        b1=theta[2*hidden_size*visible_size:2*hidden_size*visible_size+hidden_size]
        b2=theta[2*hidden_size*visible_size+hidden_size:]
        #print(W1)

        #number of training examples
        num=data.shape[1]

        #FP
        z2=W1.dot(data)+np.tile(b1,(num,1)).transpose()
        a2=sigmoid(z2)
        z3=W2.dot(a2)+np.tile(b2,(num,1)).transpose()
        h=sigmoid(z3)

        #sparsity
        rho_hat=np.sum(a2,axis=1)/num
        rho=np.tile(sparsity_param,hidden_size)

        #cost func
        cost=np.sum((h-data)**2)/(2*num)+(_lambda/2)*(np.sum(W1**2)+np.sum(W2**2))+beta*np.sum(KL_div(rho,rho_hat))

        #BP
        sparsity_delta=np.tile(-rho/rho_hat+(1 - rho)/(1 - rho_hat),(num,1)).transpose()

        delta3=-(data-h)*sigmoid_prime(z3)
        delta2=(W2.transpose().dot(delta3)+beta*sparsity_delta)*sigmoid_prime(z2)
        W1_grad=delta2.dot(data.transpose())/num+_lambda*W1
        W2_grad=delta3.dot(a2.transpose())/num+_lambda*W2
        b1_grad=np.sum(delta2,axis=1)/num
        b2_grad=np.sum(delta3,axis=1)/num


        grad=np.concatenate((W1_grad.reshape(hidden_size*visible_size),
    					W2_grad.reshape(hidden_size*visible_size),
    					b1_grad.reshape(hidden_size),
    					b2_grad.reshape(visible_size)))

        return cost,grad

