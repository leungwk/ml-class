import sys
import os
sys.path.append(os.getcwd()) # when using python-mode
import logreg # reload

import numpy as np
import scipy.optimize as scopt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import math

dat = np.genfromtxt('data/ex2data2.txt', delimiter=',')
m = dat.shape[0]
nf = dat.shape[1]-1

x1 = dat[:,0]
ux1 = np.mean(x1)
sx1 = np.std(x1)
x1 = (x1 -ux1)/(sx1)

x2 = dat[:,1]
ux2 = np.mean(x2)
sx2 = np.std(x2)
x2 = (x2 -ux2)/(sx2)

y = dat[:,2]
X = np.matrix(zip([1]*m,x1,x2)) # [1] so that an offset is uneeded

def map_feature(x1,x2):
    # deg = 6
    # data = [[1]*x1.shape[0]]
    # for i in range(1,deg+1):
    #     for j in range(i):
    #         data.append((x1**(i-j))*(x2**j))
    # return np.matrix(data).T
    return np.matrix(([1]*x1.shape[0],x1,x2,x1**2,x2**2,x1*x2)).T

Xfm = map_feature(x1,x2)

numpts = 64
x_min, x_max = x1.min(), x1.max()
y_min, y_max = x2.min(), x2.max()
hx = (x_max -x_min)/numpts
hy = (y_max -y_min)/numpts
xx, yy = np.meshgrid(np.arange(x_min, x_max, hx),
                     np.arange(y_min, y_max, hy))
Xmg = map_feature(xx.ravel(),yy.ravel())

alpha = 0.1
epsilon = 0#0.00001
niter = 10000
ldas = (0.5,1,2,4)#(0.375,0.5,0.625,0.75)
pltnum = (221,222,223,224)
lda_costs = {}

for i,lda in enumerate(ldas):
    ## with epsilon = 0, lda = 0.5, it quickly bottoms at about 0.61 after 500 iterations. scopt.optimize.fmin seems to do much better (20-40% small cost)
    costs = []
    def cost_cb(th):
        costs.append(logreg.cost(th,Xfm,y,lda))

    newths = scopt.optimize.fmin(logreg.cost,np.zeros(Xfm.shape[1]),args=(Xfm,y,lda),callback=cost_cb)
    lda_costs[lda] = costs

    # newths = np.zeros(Xfm.shape[1])
    # prev_cost = float("inf")
    # costs = []
    # for j in range(niter):
    #     new_cost = logreg.cost(newths,Xfm,y,lda)
    #     if prev_cost -new_cost <= epsilon:
    #         break
    #     costs.append(new_cost)
    #     prev_cost = new_cost

    #     newths = logreg.bgd(newths,Xfm,y,alpha,lda)
    # lda_costs[lda] = costs

    ymg = map(lambda x: 1 if logreg.sigmoid(x) >= 0.5 else 0, np.dot(Xmg,np.matrix(newths).T)) # logreg
    # ymg = map(lambda x: 1 if logreg.sigmoid(x) >= 0.5 else 0, Xmg*np.matrix(newths).T) # logreg
    ymg = np.array(ymg).reshape(xx.shape)

    plt.subplot(pltnum[i])
    plt.title("lambda = " +str(lda))
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.pcolormesh(xx, yy, ymg, cmap=plt.cm.Paired)
    plt.scatter(x1,x2,c=y,cmap=mpl.cm.get_cmap("Greens"))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# find largest x and y range
def plot_costs():
    plt.clf()
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for lda in ldas:
        max_x = max(max_x,len(lda_costs[lda]))
        max_y = max(max_y,max(lda_costs[lda]))
        min_y = min(min_y,min(lda_costs[lda]))
    for i,lda in enumerate(ldas):
        plt.subplot(pltnum[i])
        plt.title("lambda = " +str(lda))
        plt.xlabel("niter")
        plt.ylabel("cost")
        plt.plot(lda_costs[lda])

        plt.xlim(0, max_x)
        plt.ylim(min_y, max_y)
    plt.show()
