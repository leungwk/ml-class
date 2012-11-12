import sys
import os
sys.path.append(os.getcwd()) # when using python-mode
import logreg

import numpy as np
import scipy.optimize as scopt
import scipy.io as spio
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb

data = spio.loadmat("data/ex3data1.mat")
m = data['X'].shape[0]
nf = data['X'].shape[1]
X = np.matrix(np.concatenate((np.ones(m).reshape(m,-1),data['X']),axis=1))
for i in range(1,X.shape[1]):
    xs = X[:,i]
    uxs = np.mean(xs)
    sxs = np.std(xs)
    X[:,i] = (xs -uxs)/sxs if sxs != 0 else xs

y = np.matrix(data['y'])
th = np.zeros(nf+1)
lda = 1

def learn_ths():
    newths = []
    for i in [10]+range(1,9):
        tmpy = np.where(y==i,1,0).T
        newth = scopt.fmin_l_bfgs_b(logreg.cost,np.zeros(X.shape[1]),fprime=logreg.gradient,args=(X,tmpy,lda),maxfun=20,approx_grad=True,iprint=2)
        newths.append(newth)

    ## then select maxarg

    return newths

# too slow (use NNs)
# newths = learn_ths()
