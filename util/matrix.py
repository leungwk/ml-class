import numpy as np

def col_concat_ones(X):
    m = X.shape[0]
    X = np.matrix(np.concatenate((np.ones(m).reshape(m,-1),X),axis=1))
    return X
