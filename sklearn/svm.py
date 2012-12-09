import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io as spio

from sklearn import svm
from sklearn import grid_search
from sklearn import preprocessing

data = spio.loadmat("../data/ex6/ex6data2.mat")
X = data['X']
y = data['y']

def matrix_ranges(X):
    """Calculate min,max values for each column of matrix X. Return a row matrix of ranges for each column"""
    nf = X.shape[1]
    ranges = []
    for k in range(nf):
        col = X[:,k]
        ranges.append((min(col),max(col)))
    return np.matrix(ranges)

def plot_svm(X,y):
    plt.close()
    fig = plt.figure()

    numpts = 128
    rs = matrix_ranges(X)
    hx = (rs[0,1] -rs[0,0])/numpts
    hy = (rs[1,1] -rs[1,0])/numpts
    gx, gy = np.meshgrid(np.arange(rs[0,0], rs[0,1], hx),
                         np.arange(rs[1,0], rs[1,1], hy))
    grid_shape = gx.shape

    params = {'C': [0.01,0.5,1,1.2,1.5], 'gamma': [1,5,10,20,30,50]}
    svc = svm.SVC()
    clf = grid_search.GridSearchCV(svc, params)
    res = clf.fit(X, np.ravel(y))
    best = res.best_estimator_

    gxgy = zip(np.ravel(gx),np.ravel(gy))
    grid_class = best.predict(gxgy)
    grid_class = np.reshape(grid_class,grid_shape)

    ax = fig.add_subplot(111)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax.pcolormesh(gx, gy, grid_class, cmap=cmap_light)

    ax.scatter(X[:,0],X[:,1],c=y)
    ax.set_xlim(rs[0,0],rs[0,1])
    ax.set_ylim(rs[1,0],rs[1,1])

    ax.set_title("C = {0}, gamma = {1}".format(best.C,best.gamma))

    plt.show()    

def plot_Cs(X,y):
    Cs = (0.1,0.5,1,100)
    plot_pos = (221,222,223,224)

    numpts = 64#128
    rs = matrix_ranges(X)
    hx = (rs[0,1] -rs[0,0])/numpts
    hy = (rs[1,1] -rs[1,0])/numpts
    gx, gy = np.meshgrid(np.arange(rs[0,0], rs[0,1], hx),
                         np.arange(rs[1,0], rs[1,1], hy))
    grid_shape = gx.shape

    plt.close()
    fig = plt.figure()
    fig.suptitle("SVM with RPF for various C")

    # do GridSearchCV 
    for k, C in enumerate(Cs):
        clf = svm.SVC(C,gamma=10)
        clf.fit(X, np.ravel(y))
        gxgy = zip(np.ravel(gx),np.ravel(gy))
        grid_class = clf.predict(gxgy)
        grid_class = np.reshape(grid_class,grid_shape)

        ax = fig.add_subplot(plot_pos[k])
        from matplotlib.colors import ListedColormap
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        ax.pcolormesh(gx, gy, grid_class, cmap=cmap_light)

        ax.scatter(X[:,0],X[:,1],c=y)#map(lambda x: '+' if x==0 else 'o',y))
        ax.set_xlim(rs[0,0],rs[0,1])
        ax.set_ylim(rs[1,0],rs[1,1])

        ax.set_title("C = {0}".format(C))
    plt.show()

plot_svm(preprocessing.scale(X),y) # C=1, gamma=30
# plot_Cs(preprocessing.scale(X),y)

