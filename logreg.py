import numpy as np

def sigmoid(x):
    return 1./(1. +np.exp(-1.*x)) # use np rather than math for a vectorized form

def cost(th,X,y,lda):
    m = X.shape[0]
    htx = sigmoid(np.dot(X,th))
    regterm = lda/(2.*m)*np.dot(th[1:],th[1:])
    res = (1./m)*(np.dot(-1*y,np.log(htx).T) -np.dot(1 -y,np.log(1 -htx).T)) +regterm
    return res[0,0]

def bgd(th,X,y,alpha,lda):
    newths = th -alpha*gradient(th,X,y,alpha,lda)
    return np.ravel(newths.A)

def gradient(th,X,y,lda):
    m = X.shape[0]    
    htx = sigmoid(np.dot(X,th))
    return ((1./m)*(X.T*(htx -y).T).T +lda*np.r_[0,th[1:]])
