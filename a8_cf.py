import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import collections

data = spio.loadmat("data/ex8/ex8_movies.mat")
print "Loaded ex8_movies.mat"

Y = data['Y'] # ratings
R = data['R'] # "R(i,j) = 1 if and only if user j gave a rating to movie i"
n_m, n_u = Y.shape

data = spio.loadmat("data/ex8/ex8_movieParams.mat")
X = data['X']
Theta = data['Theta']
n_f = Theta.shape[1]

def J(X, Theta, Y, R, lda=0):
    "input should be arrays"
    XTheta = np.dot(X,Theta.T)
    XTheta_Y = XTheta*R -Y # each should be elementwise; R filters only for rated movies
    XTheta_Y_2 = np.power(XTheta_Y,2)
    reg_theta = 1.*lda/2*sum(sum(np.power(Theta,2)))
    reg_x = 1.*lda/2*sum(sum(np.power(X,2)))

    return 0.5*sum(sum(XTheta_Y_2)) +reg_theta +reg_x
# J(X[0:5,0:3], Theta[0:4,0:3], Y[0:5,0:4], R[0:5,0:4]) == 22.22
# J(X[0:5,0:3], Theta[0:4,0:3], Y[0:5,0:4], R[0:5,0:4], lda=1.5) == 31.34
def J_roll(XTheta, X_shape, Theta_shape, Y, R, lda=0):
    # XTheta = np.array(XTheta)
    X = XTheta[:X_shape[0]*X_shape[1]]
    Theta = XTheta[X_shape[0]*X_shape[1]:]
    res = J(X.reshape(X_shape), Theta.reshape(Theta_shape), Y, R, lda)
    return res

# grad_X and grad_Theta
def grad(X, Theta, Y, R, lda=0):
    XTheta = np.dot(X,Theta.T)
    XTheta_Y = XTheta*R -Y

    grad_X = np.dot(XTheta_Y,Theta) +lda*X
    grad_Theta = np.dot(XTheta_Y.T,X) +lda*Theta

    return (grad_X, grad_Theta)
def grad_roll(XTheta, X_shape, Theta_shape, Y, R, lda=0):
    # XTheta = np.array(XTheta)
    X = XTheta[:X_shape[0]*X_shape[1]]
    Theta = XTheta[X_shape[0]*X_shape[1]:]
    grad_X, grad_Theta = grad(X.reshape(X_shape), Theta.reshape(Theta_shape), Y, R, lda)

    grad_X_unroll = grad_X.reshape(X_shape[0]*X_shape[1],1)
    grad_Theta_unroll = grad_Theta.reshape(Theta_shape[0]*Theta_shape[1],1)
    XTheta=np.vstack((grad_X_unroll,grad_Theta_unroll))
    XTheta=np.require(XTheta,requirements='F')
    # XTheta=[x for x in np.ravel(XTheta)]
    return XTheta

# idx uses index-1
movie_list = []
with open("data/ex8/movie_ids.txt", "r") as f:
    for line in f:
        idx, name = line.split(' ', 1)
        movie_list.append((int(idx), name))
movie_list = np.rec.fromrecords(movie_list, names=["idx","names"])

my_ratings = np.zeros((n_m, 1), dtype=np.int)
my_ratings[1] = 4
my_ratings[98] = 2
my_ratings[7] = 3
my_ratings[12] = 5
my_ratings[54] = 4
my_ratings[64]= 5
my_ratings[66]= 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355]= 5
my_ratings_R = np.asarray(np.matrix(map(lambda x: 1 if x else 0, my_ratings != 0)).T)
Y = np.append(Y, my_ratings, 1)
R = np.append(R, my_ratings_R, 1)
n_m, n_u = Y.shape

## random init
X = np.random.rand(n_m, n_f)
Theta = np.random.rand(n_u, n_f)

## unroll
X_shape = X.shape
X_unroll = X.reshape(X_shape[0]*X_shape[1],1)
assert (X == X_unroll.reshape(X_shape)).all()
Theta_shape = Theta.shape
Theta_unroll = Theta.reshape(Theta_shape[0]*Theta_shape[1],1)
assert (Theta == Theta_unroll.reshape(Theta_shape)).all()

# def grad_roll(X, Theta, X_shape, Theta_Shape, Y, R, lda=0):
import scipy.optimize as spopt
XTheta=np.vstack((X_unroll,Theta_unroll))
XTheta=np.require(XTheta,requirements='F')
# XTheta=[x for x in np.ravel(XTheta)]
#
# new_XTheta,f,d = spopt.fmin_l_bfgs_b(J_roll,XTheta,grad_roll,args=(X_shape, Theta_shape, Y, R, 10), disp=1)

## save results
import cPickle as pickle
# new_X = (new_XTheta[:X_shape[0]*X_shape[1]]).reshape(X_shape)
# new_Theta = (new_XTheta[X_shape[0]*X_shape[1]:]).reshape(Theta_shape)
# with open("cache/a8_save.pickle", "w+b") as f:
#     pickle.dump([new_X, new_Theta], f)

with open("cache/a8_save.pickle", "r") as f:
    new_X, new_Theta = pickle.load(f)

P = np.dot(new_X,new_Theta.T) # prediction matrix
# compare against random predictions
# X = np.random.rand(n_m, n_f)
# Theta = np.random.rand(n_u, n_f)
# P = np.dot(X, Theta.T)

myp = P[:,-1] # most recent user added
indicies = sorted(range(len(myp)), key=lambda k: myp[k], reverse=True)
top_n = 10

print "Top {0} recommendations (out of 5):".format(top_n)
for mov_num in indicies[:top_n]:
    score = myp[mov_num]
    rec = movie_list[movie_list['idx'] == mov_num+1]
    mov_name = rec['names'][0]
    print "Predicted rating {0} for movie {1}".format(score, mov_name)

print "Original ratings provided:"
for mov_num, rat in enumerate(my_ratings):
    if rat == 0:
        continue
    rec = movie_list[movie_list['idx'] == mov_num]
    mov_name = rec['names'][0]
    print "Rated {0} for {1}".format(rat, mov_name)
