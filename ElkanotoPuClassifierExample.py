# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:38:06 2021

@author: Administrator
"""

import numpy as np

from sklearn.svm import SVC
from pulearn import WeightedElkanotoPuClassifier

import utils 

trainRate = 0.7

n_unl = 800

n_pos = 400

prior = 0.3

treshold=0.5

estimator = SVC(C=10, kernel='rbf', gamma='auto', probability=True, random_state=2018)

algorithm = "Elkanoto"

dataset = "mushroom"

if __name__ == '__main__':
    x, t = utils.load_dataset(dataset)
    xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

    xtest,ttest = utils.multiClassTest(xtest,ttest)

    P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)
    X = np.r_[P,U]
    y = np.r_[np.ones(P.shape[0]),np.zeros(U.shape[0])]
    #y = y.astype('int')
    y[np.where(y == 0)[0]] = -1.

    pu_estimator = WeightedElkanotoPuClassifier(estimator, n_pos,n_unl,hold_out_ratio=0.4)
    pu_estimator.fit(X, y)
    print(pu_estimator)
    print("\nComparison of estimator and PUAdapter(estimator):")
    print("Number of disagreements: {}".format(
        len(np.where((
            #pu_estimator.predict(X) == estimator.predict(X)
            pu_estimator.predict(xtest,treshold) == ttest
        ) == False)[0])  # noqa: E712
    ))
    print("Number of agreements: {}".format(
        len(np.where((
            pu_estimator.predict(xtest,treshold) == ttest
        ) == True)[0])  # noqa: E712
    ))
    
    