import pandas as pd
import numpy as np
import os
import gc
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import VBGMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from kaggler.online_model import SGD, FTRL, FM, NN
print 'assign path to input train/test files'
path = 'C://Users//Gaurav//Documents//EGDownloads//Kaggle//Prudential'

# Change the working directory
os.chdir(path)

print' Load train and label files'
train = pd.read_csv('mytrain_v1.csv')
#label = train.Response
label = pd.read_csv('labels.csv')
labels= pd.np.array(label)
del train['Response']
train = pd.np.array(train)
print label.shape
col = [x for x in range(8)]

X = []
X_test = []
y = []
final_res = np.zeros([label.shape[0],8])
fold = 1
for i in col:
    label = labels[:,i]
    print label.shape
    #cross_validation
    kf = KFold(train.shape[0],n_folds = 5)
    for train_indices, test_indices in kf:
        X = train[train_indices]
        y = label[train_indices]
        X_test = train[test_indices]
        X = sparse.csr_matrix(X)
        X_test = sparse.csr_matrix(X_test)
        #clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,verbose = 1)
        #clf = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto', leaf_size=30, p=1, metric='minkowski', metric_params=None)
        #clf = GaussianNB()
        #clf = OneVsRestClassifier(SVC(kernel='linear'),n_jobs = 2)
        #clf = MultinomialNB()
        #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30),n_estimators=600,learning_rate=1.5,algorithm="SAMME.R")
        #clf = VBGMM(n_components=8, covariance_type='diag', alpha=1.0, random_state=None, thresh=None, tol=0.001, verbose=1, min_covar=None, n_iter=500, params='wmc', init_params='wmc')
        print 'clf fit'
        clf = SGD(a=.01,                # learning rate
              l1=1e-6,              # L1 regularization parameter
              l2=1e-6,              # L2 regularization parameter
              n=983,              # number of hashed features
              epoch=10,             # number of epochs
              interaction=True)     # use feature interaction or not

        clf.fit(X,y)
        print 'Classifier Trained'
        #Convert the predicted array
        '''
        Y_prob = clf.predict_proba(X_test)
        Y_pred = []
        for i in range(len(Y_prob)):
                Y_pred.append([])
                for j in range(len(Y_prob[i])):
                        if len(Y_prob[i][j]) == 2:
                                Y_pred[i].append(Y_prob[i][j][1]) #positive class prob
                        else:
                                assert(len(Y_prob[i][j]) == 1)
                                prob = 1 - Y_prob[i][j][0]
                                assert(prob >= 0 and prob <= 1)
                                Y_pred[i].append(prob)
        Y_pred = np.array(Y_pred)
        print Y_pred.shape
        '''
        final_res[test_indices][i] = clf.predict(X_test)#Y_pred.transpose()
        gc.collect()
        print 'fold %d complete' %fold
        fold = fold + 1

#save res
res = pd.DataFrame(final_res)
res.to_csv('SVC_train.csv', index = False)

