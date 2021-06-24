# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:34:54 2021

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
import utils 

svc = SVC(C=1, kernel='rbf', gamma='auto', probability=True, random_state=2018)


n_unl = 2000

n_pos = 1000

prior = 0.5

trainRate = 0.7


cost_p = 1

cost_u = 0.56

x,t = utils.load_dataset("shuttle")  #1 3916 0 4208

x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)



X_train = np.r_[P, U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]

weight = [cost_p if i else cost_u for i in y_train]

svc.fit(X_train, y_train, sample_weight=weight)

y_pred = svc.predict(xtest)

errorB = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateB = errorB / ttest.shape[0]