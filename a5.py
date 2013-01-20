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
        ms.append((uxs,sxs)) # not a good idea, because it is easy to switches indicies, and it would not give a type error (should use a named tuple or dict)
        X[:,i] = (xs -uxs)/sxs if sxs != 0 else xs # warning: will modify the input X
    ms = np.matrix(ms)
    return X,ms

def unnormalize(X,ms_X):
    nf = X.shape[1]
    tmpX = np.ones((X.shape[0],1))
    for i in range(1,nf):
        tmpX = np.concatenate((tmpX,ms_X[i,1]*X[:,i] +ms_X[i,0]),axis=1)
    return tmpX[:,1:]

X,ms_X = normalize(X[:,1:]) # do not normalize 1s
X = col_concat_ones(X)
Xval,ms_Xval = normalize(Xval[:,1:])
Xval = col_concat_ones(Xval)
# note that assignment file check values are for non-normalized features, while other parts use normalized features

def train_linreg(X_in,y_in,lda):
    th = np.ones((X_in.shape[1],)) # assumed to have column of 1s
    newth,f,d = spopt.fmin_l_bfgs_b(linreg.cost,th,linreg.grad,args=(X_in,y_in,lda),m=40)
    return newth

def plot_regline(xin,yin,f):
    numpts = 96.
    outer_lim = 10 # number of steps outside of min or max

    xmin,xmax = min(xin),max(xin)
    ymin,ymax = min(yin),max(yin)
    hx = (xmax -xmin)/numpts
    hy = (ymax -ymin)/numpts
    xmin,xmax = min(xin)-outer_lim*hx,max(xin)+outer_lim*hx
    ymin,ymax = min(yin)-outer_lim*hy,max(yin)+outer_lim*hy
    hx = (xmax -xmin)/numpts
    hy = (ymax -ymin)/numpts

    xs = np.arange(xmin,xmax,hx)
    ys = map(f,xs)

    plt.scatter(xin,yin)
    plt.plot(xs,ys)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()
# newth = train_linreg(X,y,0)
# f = lambda x: newth[0] +newth[1]*x
# plot_regline(np.ravel(X[:,1]),np.ravel(y),f)




## part 2
def learning_curves(X,y,Xval,yval,lda,**keywords):
    m = X.shape[0]
    j_trains = []
    j_cvs = []
    num_trains = range(1,m+1)

    def _train(X_ts,y_ts,Xval,yval,lda):
        newth = train_linreg(X_ts,y_ts,lda)
        # j_train = linreg.cost(newth,X,y,lda=0) # incorrect, because j_train is the error of the fit of the model based on what was used to construct it (the entire training set was not used to construct it, only subsets).
        j_train = linreg.cost(newth,X_ts,y_ts,lda=0)
        j_cv = linreg.cost(newth,Xval,yval,lda)
        return (j_train,j_cv)

    for n in num_trains:
        if 'random_examples' in keywords:
            if keywords['random_examples']:
                import random
                num_times = 50
                tmp_j_train_list = []
                tmp_j_cv_list = []
                for _ in range(num_times):
                    rand_sample = random.sample(range(m),n)
                    X_ts = X[rand_sample] # X training subset
                    y_ts = y[rand_sample]
                    j_train, j_cv = _train(X_ts,y_ts,Xval,yval,lda)
                    tmp_j_train_list.append(j_train)
                    tmp_j_cv_list.append(j_cv)
                j_trains.append(np.mean(tmp_j_train_list))
                j_cvs.append(np.mean(tmp_j_cv_list))
        else:
            X_ts = X[0:n]
            y_ts = y[0:n]
            j_train, j_cv = _train(X_ts,y_ts,Xval,yval,lda)
            j_trains.append(j_train)
            j_cvs.append(j_cv)

    return (j_trains,j_cvs)

def plot_learning_curves(X,y,Xval,yval,lda,**keywords):
    j_trains,j_cvs = learning_curves(X,y,Xval,yval,lda,**keywords)
    m = X.shape[0]
    num_trains = range(1,m+1)

    plt.plot(num_trains, j_trains, '-b', label='Training')
    plt.plot(num_trains, j_cvs, '-g', label='CV')
    plt.legend(loc='upper right')
    # plt.ylim(min(min(j_trains),min(j_cvs))-1,max(max(j_trains),max(j_cvs))+1)
    plt.title('Learning curve for linear regression')
    plt.xlabel('Size of training set')
    plt.ylabel('Error')
    plt.show()

#plot_learning_curves(X,y,Xval,yval,lda=0)





## part 3
power = 8

def map_features(X,p):
    """Map onto polynormial features upto and including degree p"""
    tmpX = np.ones(X.shape)
    for k in range(p+1):
        tmpX = np.concatenate((tmpX,np.power(X,k)),axis=1)
    return tmpX[:,1:]

# X_poly_unnorm = map_features(data['X'],power)

X_poly = map_features(data['X'],power)
## normalize
X_poly,ms_X_poly = normalize(X_poly[:,1:]) # do not normalize 1s
X_poly = col_concat_ones(X_poly)

lda = 0.3 # vary this to see the effect
newth = train_linreg(X_poly,y,lda)

def finner(th,ms_X):
    def f(x):
        tmpsum = th[0]*1
        for i in range(1,th.shape[0]):
            tmpsum += th[i]*(np.power(x,i) -ms_X[i-1,0])/float(ms_X[i-1,1])
        return tmpsum
    return f

#plot_regline(np.ravel(data['X']),np.ravel(y),finner(newth,ms_X_poly)) # don't plot normalized x

X_poly_val = map_features(data['Xval'],power)
X_poly_val,ms_X_poly_val = normalize(X_poly_val[:,1:]) # do not normalize 1s
X_poly_val = col_concat_ones(X_poly_val)

X_poly_test = map_features(data['Xtest'],power)
X_poly_test,ms_X_poly_test = normalize(X_poly_test[:,1:]) # do not normalize 1s
X_poly_test = col_concat_ones(X_poly_test)

#plot_learning_curves(X_poly,y,X_poly_val,yval,lda) # shows high variance (though doesn't quite match example in assignment outline)




## part 3.3
# ldas = (0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10)
ldas = np.arange(0,10,0.1)

# uses X_poly, y, X_poly_val, yval
def plot_validation_curves(X,y,Xval,yval,ldas):
    m = X.shape[0]
    j_trains = []
    j_cvs = []
    for lda in ldas:
        newth = train_linreg(X,y,lda)
        j_train = linreg.cost(newth,X,y,lda=0)

        j_cv = linreg.cost(newth,Xval,yval,lda)

        j_trains.append(j_train)
        j_cvs.append(j_cv)

    plt.plot(ldas, j_trains, '-b', label='Training')
    plt.plot(ldas, j_cvs, '-g', label='CV')
    plt.legend(loc='upper right')
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.show()

#plot_validation_curves(X_poly,y,X_poly_val,yval,ldas) # doesn't match assignment outline

## part 3.4
print('test error: {0}'.format(linreg.cost(newth,X_poly_test,ytest,lda)))

## part 3.5
#plot_learning_curves(X_poly,y,X_poly_test,ytest,lda,random_examples=True)
