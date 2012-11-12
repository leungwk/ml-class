# import sys
# sys.path.insert(0, '../a2')
# import logreg

# capital letter for matrix, zz for list of z's, z for a scalar?

import os

import copy
import numpy as np
import scipy.optimize as spopt
import scipy.io as spio
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import itertools

"""
Neural net for digit recognition
"""

data = spio.loadmat("data/ex4data1.mat")
m = data['X'].shape[0]
nf = data['X'].shape[1]
X = np.matrix(np.concatenate((np.ones(m).reshape(m,-1),data['X']),axis=1))
## is feature normalization needed with NNs?
for i in range(1,X.shape[1]): # do not normalize 1s
    xs = X[:,i]
    uxs = np.mean(xs)
    sxs = np.std(xs)
    X[:,i] = (xs -uxs)/sxs if sxs != 0 else xs

## turn labels (1,2,3,...,8,9,0) into 10-D vecs (with [0,0,...,0,1] as digit '0')
y = np.matrix(data['y'])
nout = 10
tmpys = []
for i in range(m):
    tmp = np.zeros(nout)
    tmp[y[i] -1] = 1
    tmpys.append(tmp)
Y = np.matrix(tmpys)



th = np.zeros(nf+1)
lda = 1

prew = spio.loadmat("data/ex4weights.mat")
Theta1 = prew['Theta1']
Theta2 = prew['Theta2']

TH = [Theta1,Theta2] # generally initalize with small +/- \epsilon according to nn arch


TH = []
nnarch = [(25, 401),(10, 26)] # first number is # units in layer l+1
lin = nf
lout = 0
## init weights
for s in nnarch:
    lout = s[0]
    einit = np.sqrt(6)/np.sqrt(lin +lout)
    TH.append(np.random.uniform(-1.*einit,einit,s))
    lin = lout










# confusion matrix graphic: http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
# http://stackoverflow.com/a/10958893
#
def confusion_matrix(ytrue,ypred): # must be row vectors
    classes = list(set(ytrue.flat))
    n = len(classes)

    return np.bincount(np.array(n * (ytrue -1) +(ypred -1))[0], minlength=n*n).reshape(n, n)

def err(X,y,TH):
    m = X.shape[0]
    htx = ff(X,TH)[-1]
    ynn = htx.argmax(axis=1) +1 # if 10th el is max, then '0'
    cfmtx = confusion_matrix(y.T,ynn.T)
    return 1 -float(sum(np.diag(cfmtx)))/float(m)






def sigmoid(X):
    return 1./(1. +np.exp(-1.*X)) # use np rather than math for a vectorized form

def sigmoid_grad(X):
    return np.multiply(sigmoid(X),(1. -sigmoid(X)))

def ff(X,TH):
    m = X.shape[0]
    a = X
    aa = [a]
    for th in TH:
        z = a*(th.T) #np.dot(a,th.T)
        a = sigmoid(z)
        a = np.matrix(np.concatenate((np.ones(m).reshape(m,-1),a),axis=1)) # augment with 1s

        aa.append(a)
    htx = a[:,1:]
    aa[-1] = htx # remove '+1' unit from output layer

    # htx==a^{(L)} (without '+1' unit)
    # a^{(1)},a^{(2)},...,a^{(L -1)}
    return aa

def cost_wrap_l_bfgs_b(linTH,X,Y,lda,dims):
    """th is an array to be reshaped by each dims"""
    TH = roll(linTH,dims)
    return cost(TH,X,Y,lda)

def cost(TH,X,Y,lda=1):
    """TH is rolled (orderedly collected) with Th1, Th2, ... """
    m = X.shape[0]
    htx = ff(X,TH)[-1]

    termsum = 0
    for rn in range(m):
        termsum += (-1.*Y[rn,:])*(np.log(htx[rn,:]).T) -(1. -Y[rn,:])*(np.log(1. -htx[rn,:]).T)
    termsum = 1./m*(termsum)

    regsum = 0
    for th in TH:
        regsum += sum(sum(np.power(th[:,1:],2)))
    regsum = lda/(2.*m)*regsum

    return termsum[0,0] +regsum

def bp(aa,Y,TH):
    m = Y.shape[0]
    L = len(TH) +1

    deltaL = aa[-1] -Y # htx == aa[-1]
    dd = [deltaL]
    for l in range(L -1, L -1 -1, -1):
        delta_prev = dd[-1] # as in one layer ahead

        ## remove delta for bias unit
        delta = np.multiply(delta_prev*(TH[l -1][:,1:]), # backwards? what if delta_prev were a middle layer? shouldn't it's delta_0 be removed, not TH[l -1][0]?
                            np.multiply(sigmoid(aa[l -1][:,1:]),
                                        (1. -sigmoid(aa[l -1][:,1:]))))
        dd.append(delta)

        delta_prev = delta
    dd = [1]+dd[::-1] # reverse list $1,\delta^{(2)},\delta^{(3)}$

    # something is wrong because for TH[1] the grad diff is too large
    # floating point losses?
    DD = []
    for l in range(L -1):
        tmpD = np.zeros(TH[l].shape)
        for mi in range(m):
            tmpa = aa[l][mi,:]
            tmpd = dd[l +1][mi,:]
            tmpD = tmpD +(tmpd.T)*tmpa
        Delta = 1./m*tmpD

        DD.append(Delta)

    return DD

def grad_wrap_l_bfgs_b(linTH,X,Y,lda,dims):
    """th is an array to be reshaped by each dims"""
    TH = roll(linTH,dims)

    # unravel
    G = grad(TH,X,Y,lda)
    linG = np.array(())
    for g in G:
        linG = np.concatenate((linG,np.array(g.ravel())[0]))
    return linG

def grad(TH,X,Y,lda=1):
    m = X.shape[0]
    aa = ff(X,TH)
    GG = bp(aa,Y,TH)
    for l,gg in enumerate(GG):
        GG[l][:,1:] = gg[:,1:] +lda/float(m)*TH[l][:,1:]
    return GG

def grad_check(X,Y,TH):
    epsilon = 10**(-4)
    THp = copy.copy(TH)
    THn = copy.copy(TH)
    lda = 0.3
    DD = grad(TH,X,Y,lda)

    f = open('grad_check.txt', 'w+')
    for ldx,(th,dd) in enumerate(zip(TH,DD)):
        th = th[:,1:] # remove weight from bias term
        for j in range(th.shape[1]):
            for i in range(th.shape[0]):
                THp[ldx][i,j] = THp[ldx][i,j] +epsilon
                THn[ldx][i,j] = THn[ldx][i,j] -epsilon

                tmp2 = dd[i,j]
                tmp = (cost(THp,X,Y,lda) -cost(THn,X,Y,lda))/(2.*epsilon)
                diff = np.abs(tmp2 -tmp)
                # if diff >= 1E-4:
                f.write("TH[" +str(ldx) +"][" +str(i) +"," +str(j) +"] diff is " +str(diff) +"\n")

                THp[ldx][i,j] = THp[ldx][i,j] -epsilon
                THn[ldx][i,j] = THn[ldx][i,j] +epsilon

            ## force write of buffer
            f.flush()
            os.fsync(f)
    f.close()

def unroll(TH):
    linTH = np.array(())
    dims = []
    for th in TH:
        dims.append(th.shape)
        linTH = np.concatenate((linTH,th.ravel()))
    return linTH,dims
def roll(linx,dims):
    lb = 0
    TH = []
    for dim in dims:
        nel = dim[0]*dim[1]
        ub = lb+nel
        ths = linx[lb:ub].reshape(dim)
        TH.append(ths)
        lb = ub
    return TH

linTH,dims = unroll(TH)
THmin = roll(linTH,dims)

# newTH,f,d = spopt.fmin_l_bfgs_b(cost_wrap_l_bfgs_b,linTH,grad_wrap_l_bfgs_b,args=(X,Y,1,dims),iprint=2,maxfun=400,m=40) # if it seems like the cost is still too high, try increasing m
# newTHmin = roll(newTH,dims)
# errnn = err(X,y,newTHmin)
"""
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N   Tit  Tnf  Tnint  Skip  Nact     Projg        F
10285   41  361     10     1     0   6.216D-03   6.982D-01
  F =  0.69818676720766581     

ABNORMAL_TERMINATION_IN_LNSRCH                              

 Line search cannot locate an adequate point after 20 function
  and gradient evaluations.  Previous x, f and g restored.
 Possible causes: 1 error in function or gradient evaluation;
                  2 rounding error dominate computation.

 Cauchy                time 3.906E-03 seconds.
 Subspace minimization time 6.409E-02 seconds.
 Line search           time 7.858E+02 seconds.

 Total User time 8.343E+02 seconds.

>>> newTHmin = roll(newTH,dims)
newTHmin = roll(newTH,dims)
>>> errnn = err(X,y,newTHmin)
>>> errnn
0.0928
"""



# grad_check(X[::250],Y[::250],TH)
# cst = cost(TH,X,Y,lda=1)
# DD = grad(TH,X,Y,lda=1)
