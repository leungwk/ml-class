import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import math

"""
Implements linear regression with batch gradient descent
"""

dat = np.genfromtxt('data/ex1data1.txt', delimiter=',')
m = dat.shape[0]
nf = dat.shape[1]-1
xs = dat[:,0]
ys = dat[:,1]
X = np.matrix(zip([1]*m,xs))

itcnt = 1500
alpha = 0.01
ths = np.array([0]*X.shape[1])

def cost(X,ys,ths):
    m = X.shape[0]
    tmpsum = 0
    for i in range(m):
        xi = np.ravel(X[i,:])
        yi = ys[i]
        h_xi = sum(xi*ths)
        tmpsum += (h_xi -yi)**2

    return (1/(2.*m))*tmpsum

def bgd(X,ys,ths,alpha,itcnt):
    """
    First column of X should be all 1 
    """
    m = X.shape[0]
    nf = X.shape[1]
    
    newths = [-1]*len(ths)
    if len(ths) != nf:
        raise Exception("Length of theta vector not equal to number of features")

    costs = []
    for niter in range(itcnt):
        costs.append(cost(X,ys,ths))

        ## simulatenous update
        for j,th in enumerate(ths):
            tmpsum = 0
            for i in range(m):
                xi = np.ravel(X[i,:])
                yi = ys[i]
                h_xi = sum(xi*ths)
                tmpsum += (h_xi -yi)*xi[j]
            newths[j] = th -(alpha/m)*tmpsum
        ths = newths

    return costs,newths

costs,newths = bgd(X,ys,ths,alpha,itcnt)

plt.scatter(xs,ys)
lfxs = range(int(math.ceil(min(xs))),int(math.floor(max(xs))))
lfys = map(lambda x: newths[0] +newths[1]*x, lfxs)
plt.plot(lfxs,lfys)
plt.show()
