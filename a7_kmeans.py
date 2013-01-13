import numpy as np
import scipy.io as spio
import scipy.misc as spmisc
import random
import matplotlib.pyplot as plt

data = spio.loadmat("data/ex7/ex7data2.mat")
X = data['X']

def cost(X, MU, C):
    m, nf = X.shape
    total_cost = 0
    for i, x in enumerate(X):
        mu = MU[C[i]]
        total_cost += np.power(np.linalg.norm(x-mu),2)
    total_cost = 1./m*total_cost
    return total_cost

def kmeans(X, k, epsilon=1e-4, niter=1000):
    m, nf = X.shape

    ## select initial means
    C = X[random.sample(range(m),k)]
    # C = np.matrix([[3,3],[6,2],[8,5]])

    costs = []
    for iter_num in range(niter):
        ## assign each example to a centroid
        C_idx = [] # centroid indicies for each training example
        for x in X:
            vals = [np.power(np.linalg.norm(x-mu),2) for mu in C]
            C_idx.append(vals.index(min(vals)))
        C_idx = np.array(C_idx)

        ## update centroids
        C_new = []
        for i in range(k):
            Xi_idxs = np.where(C_idx == i)[0]
            Xi_cnt = len(Xi_idxs)
            C_new.append(1./Xi_cnt*np.sum(X[Xi_idxs], axis=0) if Xi_cnt > 0 else X[random.randint(0,m-1)])
        C = C_new = np.array(C_new)

        new_cost = cost(X,C_new,C_idx)
        if costs:
            old_cost = costs[-1]
            if np.abs(new_cost -old_cost) < epsilon:
                costs.append(new_cost)
                break
            else:
                costs.append(new_cost)
        else:
            costs.append(new_cost)

    return C_new, C_idx, np.array(costs)

def run_kmeans(num_runs = 100):
    C_best, C_idx_best, cost_best = None, None, float("inf")
    for _ in range(num_runs):
        C_new, C_idx, _ = kmeans(X, 3)
        new_cost = cost(X,C_new,C_idx)
        if new_cost < cost_best:
            C_best, C_idx_best, cost_best = C_new, C_idx, new_cost
    return C_best, C_idx_best, cost_best

# C_best, C_idx_best, cost_best = run_kmeans(100)
# plot_kmeans(X, C_best, C_idx_best)
def plot_kmeans(X, C, C_idx):
    plt.close()
    plt.scatter(X[:,0], X[:,1], c=C_idx)
    plt.scatter(C[:,0], C[:,1], s=192, marker='*')
    plt.show()





bird_data = spmisc.imread("data/ex7/bird_small.png")
bird_data_shape = bird_data.shape

img_data = np.reshape(bird_data, (bird_data_shape[0]*bird_data_shape[1], 3))
img_C, img_C_idx, img_costs = kmeans(img_data, 16, niter=10)

img_list = [img_C[ci] for ci in img_C_idx]
img_recovered = np.array(img_list)
img_recovered = np.reshape(img_recovered, bird_data_shape)
fname = "data/ex7/output_ex7.png"
spmisc.imsave(fname, img_recovered)
print "Saved output to {0}".format(fname)
