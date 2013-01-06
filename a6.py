import numpy as np
import matplotlib.pyplot as plt
import re
import nltk.stem.porter
from nltk.tokenize import wordpunct_tokenize, sent_tokenize

vocab_list = np.genfromtxt('data/ex6/vocab.txt', delimiter='\t', dtype=[("idx", int), ("word", object)], converters={1: str})
# vocab_list = np.array((vocab['idx'], map(str, vocab['word'])))
vocab_dict = dict()
for idx, word in vocab: # idx and word are two different types
    vocab_dict[word] = idx
    # vocab_dict[idx] = word

email_fname = 'data/ex6/emailSample1.txt'
def process_email(email_fname, vocab_dict):
    lines = []
    with open(email_fname) as f:
        lines = f.readlines()
    lines = [line.lower() for line in lines]
    lines = [re.sub('<[^<>]+>', ' ', line) for line in lines]
    lines = [re.sub('[0-9]+', 'number', line) for line in lines]
    lines = [re.sub('(http|https)://[^\s]*', 'httpaddr', line) for line in lines]
    lines = [re.sub('[^\s]+@[^\s]+', 'emailaddr', line) for line in lines]
    lines = [re.sub('[$]+', 'dollar', line) for line in lines]

    lines = [re.sub('[@$/#\.-:&\*\+=\[\]\?!\(\)\{\},\'">_<;%\n]', '', line) for line in lines]
    line = " ".join(lines)
    words = wordpunct_tokenize(line)
    
    ps = nltk.stem.porter.PorterStemmer()
    words = [ps.stem(w) for w in words]
    
    words_idx_list = []
    for word in words:
        if word in vocab_dict:
            words_idx_list.append(vocab_dict[word])
    return words_idx_list
        
email_word_indicies = process_email(email_fname, vocab_dict)

def email_features(word_indicies, vocab_list):
    feature_list = [] # size equal to vocab
    word_indicies_set = set(word_indicies)
    for idx, _ in vocab_list:
        feature_list.append(1 if idx in word_indicies_set else 0)
    return feature_list
            
res = email_features(email_word_indicies, vocab_list)

## apparently don't have to implement a SVM, just use an existing one ...

from sklearn import svm
from sklearn import grid_search
import scipy.io as spio
train_data = spio.loadmat("data/ex6/spamTrain.mat")
X_train = train_data['X']
y_train = train_data['y']
n_train = len(y_train)

test_data = spio.loadmat("data/ex6/spamTest.mat")
X_test = test_data['Xtest']
y_test = test_data['ytest']
n_test = len(y_test)

# should have a cv set ...

# C = 0.1
# use linear kernel
svc = svm.SVC()#(C, kernel='linear')
params = {'C': [0.01,0.05,0.1,0.25,0.5,0.75,1,1.25,1.5],
          # 'gamma': [1,5,10,20,30,50],
          'kernel':['linear']}
clf = grid_search.GridSearchCV(svc, params)
res = clf.fit(X_train, np.ravel(y_train))
clf = res.best_estimator_ # SVC(C=0.05, cache_size=200, coef0=0.0, degree=3, gamma=0.0, kernel='linear', probability=False, scale_C=False, shrinking=True, tol=0.001)

y_train_predict = clf.predict(X_train)
print "Training accuracy is {0}".format(1.0*list(np.ravel(y_train) -y_train_predict).count(0)/n_train)

y_test_predict = clf.predict(X_test)
print "Test accuracy is {0}".format(1.0*list(np.ravel(y_test) -y_test_predict).count(0)/n_test)

## get top predictors for spam
top_feature_indicies = (np.ravel(np.argsort(np.ravel(clf.coef_))) -1)[::-1] # .reverse() # np.array doesn't have reverse feature
top_n = 14
for i in range(top_n):
    tfi = top_feature_indicies[i]
    print vocab_list[tfi]
