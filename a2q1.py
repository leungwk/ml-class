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

dat = np.genfromtxt('data/ex2data1.txt', delimiter=',')
m = dat.shape[0]
nf = dat.shape[1]-1

# I couldn't believe this makes bgd go faster. To see why, don't do feature standardization, and run bgd. If the data has "large" numbers (ie. >> 1), alpha will be very sensitive.
x1 = dat[:,0]
ux1 = np.mean(x1)
sx1 = np.std(x1)
x1 = (x1 -ux1)/(sx1)

x2 = dat[:,1]
ux2 = np.mean(x2)
sx2 = np.std(x2)
x2 = (x2 -ux2)/(sx2)

y = dat[:,2]
X = np.matrix(zip([1]*m,x1,x2)) # always augment X with \vec{1} after feature standarization
th = np.zeros(3)

# skip gradient descent or fminunc
#newths = scopt.optimize.fmin(logreg.cost,th,args=(X,y,0))

lda = 0
epsilon = 0.0001
niter = 10000

alphas = (0.1,0.5,1,2)
pltnum = (221,222,223,224)
alpha_costs = {}
for alpha in alphas:
    costs = []
    newths = th
    prev_cost = float("inf")
    for i in range(niter):
        new_cost = logreg.cost(newths,X,y,lda)
        if prev_cost -new_cost <= epsilon:
            break
        costs.append(new_cost)
        prev_cost = new_cost

        newths = logreg.bgd(newths,X,y,alpha,lda)
    costs = np.array(costs)

    alpha_costs[alpha] = costs

numpts = 64
x_min, x_max = x1.min(), x1.max()
y_min, y_max = x2.min(), x2.max()
hx = (x_max -x_min)/numpts
hy = (y_max -y_min)/numpts
xx, yy = np.meshgrid(np.arange(x_min, x_max, hx),
                     np.arange(y_min, y_max, hy))

db = lambda x1: (-1./newths[2])*(newths[1]*x1 +newths[0])
dbxs = np.arange(x_min, x_max, hx)
dbys = map(db, dbxs)

## visualize data and boundary
plt.scatter(x1,x2,c=map(int,y),cmap=mpl.cm.get_cmap("Greens"))
plt.plot(dbxs,dbys,ls='-')
# plt.show()

# find largest x and y range
max_x = float("-inf")
min_y = float("inf")
max_y = float("-inf")
for alpha in alphas:
    max_x = max(max_x,len(alpha_costs[alpha]))
    max_y = max(max_y,max(alpha_costs[alpha]))
    min_y = min(min_y,min(alpha_costs[alpha]))

# plt.clf(); plt.plot(costs); plt.show()
for i,alpha in enumerate(alphas):
    plt.subplot(pltnum[i])
    plt.title("alpha = " +str(alpha))
    plt.xlabel("niter")
    plt.ylabel("cost")
    plt.plot(alpha_costs[alpha])

    # plt.ylim(min_y, max_y)
    plt.xlim(0, max_x)
