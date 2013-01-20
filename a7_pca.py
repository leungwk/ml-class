import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from matplotlib.figure import Figure                       
from matplotlib.axes import Axes                           
from matplotlib.lines import Line2D

data = spio.loadmat("data/ex7/ex7data1.mat")
X = np.matrix(data['X'])

def normalize(X):
    nf = X.shape[1]
    ms = [] # mean and standard deviation
    for i in range(nf):
        xs = X[:,i]
        uxs = np.mean(xs)
        sxs = np.std(xs)
        ms.append((uxs,sxs)) # not a good idea, because it is easy to switches indicies, and it would not give a type error (should use a named tuple or dict)
        X[:,i] = (xs -uxs)/sxs if sxs != 0 else xs
    ms = np.matrix(ms)
    return X,ms

def pca(X):
    Sigma = np.cov(Xnorm, rowvar=0) # each row is a variable
    # from the manual: "The rows of v are the eigenvectors of a.H a. The columns of u are the eigenvectors of a a.H. For row i in v and column i in u, the corresponding eigenvalue is s[i]**2.", where a is the input
    # D scales, V rotates, and U is a perfect circle
    U, S, V = np.linalg.svd(Sigma.newbyteorder('=')) # newbyteorder is workaround
    return np.matrix(U), S, np.matrix(V)

def recover_data(Z, U):
    pass

Xnorm, ms = normalize(X)
X = np.matrix(data['X']) # because destructive normalize
U, S, V = pca(Xnorm)

num_dim = 1
Ureduce = U[:,:num_dim]
Z = np.dot(Xnorm,Ureduce) # is this "stretching", for each point in Xnorm, along the eigenvectors (columns of Ureduce)? should be since Ureduce has eigenvectors, making Xnorm a "stretching" only

Xproj = Z*Ureduce.T

def plot_pca_proj():
    plt.close()
    plt.clf()
    # plt.scatter(np.ravel(X[:,0]),np.ravel(X[:,1]),c='b') # gives slightly off plots if a matrix if a 1D matrix is passed in ...
    plt.scatter(np.ravel(Xnorm[:,0]),np.ravel(Xnorm[:,1]),c='b')
    plt.scatter(np.ravel(Xproj[:,0]),np.ravel(Xproj[:,1]),c='r')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.show()

# plot_pca_proj()

data = spio.loadmat("data/ex7/ex7faces.mat")
X = np.matrix(data['X'])
Xnorm, ms = normalize(X)
X = np.matrix(data['X'])
U, S, V = pca(Xnorm)


