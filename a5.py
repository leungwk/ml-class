import sys
import os
sys.path.append(os.getcwd()) # when using python-mode
import linreg; reload(linreg)

import numpy as np
import scipy.io as spio
import scipy.optimize as spopt
import matplotlib.pyplot as plt

data = spio.loadmat("data/ex5data1.mat")
m = data['X'].shape[0]
nf = data['X'].shape[1]

X = np.matrix(np.concatenate((np.ones(m).reshape(m,-1),data['X']),axis=1))
y = data['y']
Xtest = data['Xtest']
Xval = data['Xval']
ytest = data['ytest']

# normalize features
ms = [] # mean and standard deviation
for i in range(1,X.shape[1]): # do not normalize 1s
    xs = X[:,i]
    uxs = np.mean(xs)
    sxs = np.std(xs)
    ms.append((uxs,sxs))
    X[:,i] = (xs -uxs)/sxs if sxs != 0 else xs
ms = np.matrix(ms)
# note that assignment file check values are for non-normalized features

th = np.ones((1+nf,))

newth,f,d = spopt.fmin_l_bfgs_b(linreg.cost,th,linreg.grad,args=(X,y,1),maxfun=400,m=40)

def plot_regline(xin,yin,th):
    numpts = 64.
    xmin,xmax = min(xin),max(xin)
    hx = (xmax -xmin)/numpts

    xs = np.arange(xmin,xmax,hx)
    ys = map(lambda x: th[0] +th[1]*x,xs)

    plt.scatter(xin,yin)
    plt.plot(xs,ys)
    plt.show()

plot_regline(np.ravel(X[:,1]),np.ravel(y),newth)
