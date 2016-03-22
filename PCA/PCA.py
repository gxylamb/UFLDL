import numpy as np
import matplotlib.pyplot as plt
import math

#from data file
#2D data in one row
def getMean(data):
	meanVal=np.mean(data,axis=1)
	meanVal=np.reshape(meanVal,(2,1))
	_data=data-meanVal	
	return _data,meanVal

data=np.loadtxt('data.txt')
plt.scatter(data[0],data[1])
plt.show()
#==================================================================================
_data,meanVal=getMean(data)
#rowvar should be non-zero(default)
#so here is nonsense
#but i am gonna keep it
covMat=np.cov(_data,rowvar=1)
eigVal,eigVec=np.linalg.eig(np.mat(covMat))
#print(eigVal,eigVec)
plt.plot([0,eigVec[0,0]],[0,eigVec[1,0]],color="blue",linewidth=1.0,linestyle='-')
plt.plot([0,eigVec[0,1]],[0,eigVec[1,1]],color="blue",linewidth=1.0,linestyle='-')
plt.show()
#==================================================================================
#get xrot
print(eigVec)
eigValIndice=np.argsort(eigVal)
#sort from small to big and use below to reverse
# [::-1] [::-1] [::-1] [::-1] WTF????????
eigValIndice=eigValIndice[::-1]
eigVec=eigVec[:,eigValIndice]
xRot=np.dot(eigVec.T,_data)
plt.scatter(xRot[0],xRot[1])
plt.show()
#==================================================================================
U=eigVec.copy();
#Then we lower the dimension 
#by assign the others with 0 
U[:,1]=[0]
xHat=np.dot(U,xRot)
plt.scatter(xHat[0],xHat[1])
plt.show()
#==================================================================================
#Whitening
epsilon=10e-5
eigVal=eigVal+epsilon
eigVal=eigVal.reshape(len(eigVal),1)
eigVal=eigVal[eigValIndice,:]
xPcaWhite=xRot/np.sqrt(eigVal)
#ZCA
xZcaWhite=np.dot(eigVec,xPcaWhite)
plt.scatter(xZcaWhite[0],xZcaWhite[1])
plt.show()
#==================================================================================




