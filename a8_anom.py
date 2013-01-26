import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import collections

data = spio.loadmat("data/ex8/ex8data1.mat")
print "Loaded ex8data1.mat"
X = data['X']
# for validation
Xval = data['Xval']
yval = data['yval']
Mu = np.array(np.matrix(np.mean(X, axis=0)).T) # to treat as a column vector
Sigma = np.cov(X, rowvar=0)

def plot_x(X):
    plt.close()
    plt.scatter(X[:,0], X[:,1])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()

def mvn(x, Mu, Sigma):
    n = len(Mu)
    if Mu.shape != x.shape:
        raise Exception("Mu and x have different shapes")
    xminusmu = x-Mu
    return (1./np.sqrt(np.power((2*np.pi),n)*np.linalg.det(Sigma))*np.exp(-0.5*(xminusmu.T)*(np.linalg.inv(Sigma))*xminusmu))[0,0] # why does (Sigma**(-1)) not work?
# mvn = np.vectorize(mvn)
def pmvn(X, Mu, Sigma):
    pval = []
    for row in X:
        pval.append(mvn(np.matrix([row]).T, Mu, Sigma))
    pval = np.ravel(np.array(pval))
    return pval


def confusion_matrix(ytrue,ypred):
    """src: http://stackoverflow.com/a/10958893"""
    import itertools
    classes = list(set(ytrue))
    n = len(classes)
    return np.array([zip(ytrue,ypred).count(x) for x in itertools.product(classes,repeat=2)]).reshape(n,n)
# confusion_matrix([0,0,0,0,1,1,1,1,1,1,1],
#                  [0,1,1,1,1,1,0,0,1,1,1])
# array([[1, 3],   [tn, fp]
#       [2, 5]])   [fn, tp]

def select_threshold(yval, pval, n=2):
    es = np.array([k*10**(-p) for p in range(1,30) for k in range(1,10)])
    best_epsilon = None
    best_f1 = -float("inf")
    f1s = []
    for epsilon in es:
        anoms = np.matrix(map(lambda x: 1 if x else 0, pval < epsilon)).T
        cmtx = confusion_matrix(list(yval.flat), list(anoms.flat))
        tn = cmtx[0,0]
        fp = cmtx[0,1]
        fn = cmtx[1,0]
        tp = cmtx[1,1]
        prec = 1.*tp/(tp+fp)
        rec = 1.*tp/(tp+fn)
        f1 = 2.*prec*rec/(prec+rec)
        f1s.append(f1)
        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
    return (best_epsilon, best_f1, (es,f1s))

pval = pmvn(Xval, Mu, Sigma)
best_epsilon, best_f1, _ = select_threshold(yval, pval)
print "best_epsilon: {0} (F1 = {1})".format(best_epsilon, best_f1)

def plot_anom(X, epsilon):
    px = []
    for row in X:
        px.append(mvn(np.matrix([row]).T, Mu, Sigma))
    px = np.ravel(np.array(px))

    anoms = px < epsilon
    print "anom_count: {0}".format(sum(map(lambda x: 1 if x else 0, anoms)))

    plt.scatter(X[:,0], X[:,1], c=map(lambda x: 'r' if x else 'b', anoms))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.title('Anomaly detection')
# plt.close()
plot_anom(X, best_epsilon)
# plt.show()


data = spio.loadmat("data/ex8/ex8data2.mat")
print "Loaded ex8data2.mat"
X = data['X']
# for validation
Xval = data['Xval']
yval = data['yval']
Mu = np.array(np.matrix(np.mean(X, axis=0)).T) # to treat as a column vector
Sigma = np.cov(X, rowvar=0)

pval = pmvn(Xval, Mu, Sigma)
best_epsilon, best_f1, (es, f1s) = select_threshold(yval, pval, n=Xval.shape[1])
print "best_epsilon: {0} (F1 = {1})".format(best_epsilon, best_f1)

# plt.close()
plot_anom(X, best_epsilon)
# plt.show()
