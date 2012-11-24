import sys
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd()) # for when using python-mode
import linreg; reload(linreg)

import numpy as np
import scipy.io as spio
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from util.matrix import col_concat_ones # requires that __init__.py exist under util

data = spio.loadmat("data/ex5data1.mat")
m = data['X'].shape[0]
nf = data['X'].shape[1]

X = col_concat_ones(data['X'])
y = data['y']
mval = data['Xval'].shape[0]
Xval = col_concat_ones(data['Xval'])
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

def normalize(X):
    nf = X.shape[1]
    ms = [] # mean and standard deviation
    for i in range(nf):
        xs = X[:,i]
        uxs = np.mean(xs)
        sxs = np.std(xs)
        ms.append((uxs,sxs))
        X[:,i] = (xs -uxs)/sxs if sxs != 0 else xs
    ms = np.matrix(ms)
    return X,ms

X,ms_X = normalize(X[:,1:]) # do not normalize 1s
X = col_concat_ones(X)
Xval,ms_Xval = normalize(Xval[:,1:])
Xval = col_concat_ones(Xval)
# note that assignment file check values are for non-normalized features, while other parts use normalized features

th = np.ones((1+nf,))

def train_linreg(X_in,y_in,lda):
    newth,f,d = spopt.fmin_l_bfgs_b(linreg.cost,th,linreg.grad,args=(X_in,y_in,lda),maxfun=400,m=40)
    return newth

def plot_regline(xin,yin,th):
    numpts = 64.
    xmin,xmax = min(xin),max(xin)
    hx = (xmax -xmin)/numpts

    xs = np.arange(xmin,xmax,hx)
    ys = map(lambda x: th[0] +th[1]*x,xs)

    plt.scatter(xin,yin)
    plt.plot(xs,ys)
    plt.show()

#plot_regline(np.ravel(X[:,1]),np.ravel(y),train_linreg(X,y,0))

# part 2
def plot_learning_curves(X_in,y_in,Xval_in,yval_in):
    j_trains = []
    j_cvs = []
    num_trains = range(1,m+1)
    for n in num_trains:
        X_ts = X_in[0:n] # X training subset
        y_ts = y_in[0:n]
        lda = 0

        newth = train_linreg(X_ts,y_ts,lda)
        # j_train = linreg.cost(newth,X_in,y_in,lda=0) # incorrect, because j_train is the error of the fit of the model based on what was used to construct it (the entire training set was not used to construct it, only subsets).
        j_train = linreg.cost(newth,X_ts,y_ts,lda=0)

        j_cv = linreg.cost(newth,Xval_in,yval_in,lda)

        j_trains.append(j_train)
        j_cvs.append(j_cv)

    plt.plot(num_trains, j_trains, '-b', label='Training')
    plt.plot(num_trains, j_cvs, '-g', label='CV')
    plt.legend(loc='upper right')
    plt.title('Learning curve for linear regression')
    plt.xlabel('Size of training set')
    plt.ylabel('Error')
    plt.show()

#plot_learning_curves(X,y,Xval,yval)
