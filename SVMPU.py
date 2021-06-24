# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:41:20 2021

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
import utils 

svc = SVC(C=1, kernel='rbf', gamma='auto', probability=True, random_state=2018)


n_unl = 500

n_pos = 200

prior = 0.5

trainRate = 0.7

x,t = utils.load_dataset("banknote")  #1 3916 0 4208
#mushroom 1 3916 0 4208
#banknote 1 762 0 610
#optdigits 1 572 0 5048
#spambase 1 1813 0 2788  C=100 lamda 0.015
#page-blocks1 4913  0 560
#diabetes 1 268 0 500
#balance 1 288 0 337
#credit 1 700 0 300
#madelon 1 1300 0 1300
#house 1 8914 0 11726
#wdbc 1 357 0 212
#solet 1 300 0 300

x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)

X_train = np.r_[P, U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]

svc.fit(X_train, y_train)
y_pred = svc.predict(xtest)

errorSVMPU = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateSVMPU = errorSVMPU / ttest.shape[0]