import numpy as np

def cost(th,X,y,lda):
    m = X.shape[0]

    htx = np.dot(X,th) # elementwise, returns row vec
    tmpterm = htx -y
    ssterm = 1./(2.*m)*np.dot(tmpterm.T,tmpterm)

    regterm = lda/(2.*m)*np.dot(th[1:].T,th[1:])

    return (ssterm +regterm)[0,0]

def grad(th,X,y,lda):
    m = X.shape[0]
    htx = np.dot(X,th) # elementwise
    htxmy = htx.T -y
    sumterm = (1./m*(htxmy.T*X))

    gradterm = sumterm +np.matrix(lda/m*np.concatenate((np.zeros(1),th[1:])))
    return np.ravel(gradterm)
