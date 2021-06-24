# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:38:06 2021

@author: Administrator
"""

import numpy as np

from sklearn.svm import SVC
from pulearn import ElkanotoPuClassifier

import utils 

trainRate = 0.9

n_unl = 400

n_pos = 200

prior = 0.3

estimator = SVC(C=10, kernel='rbf', gamma='auto', probability=True, random_state=2018)

if __name__ == '__main__':
    x, t = utils.load_dataset("banknote")
    xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

    xtest,ttest = utils.multiClassTest(xtest,ttest)

    P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)
    X = np.r_[P,U]
    y = np.r_[np.ones(P.shape[0]),np.zeros(U.shape[0])]
    #y = y.astype('int')
    y[np.where(y == 0)[0]] = -1.

    pu_estimator = ElkanotoPuClassifier(estimator, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)
    print(pu_estimator)
    print("\nComparison of estimator and PUAdapter(estimator):")
    print("Number of disagreements: {}".format(
        len(np.where((
            pu_estimator.predict(X) == estimator.predict(X)
        ) == False)[0])  # noqa: E712
    ))
    print("Number of agreements: {}".format(
        len(np.where((
            pu_estimator.predict(X) == estimator.predict(X)
        ) == True)[0])  # noqa: E712
    ))
    
    