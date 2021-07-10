# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:29:41 2021

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
import utils 

svc = SVC(C=1, kernel='rbf', gamma='auto', probability=True, random_state=2018)



trainRate = 0.7

testRate = 0.99



x,t = utils.load_dataset("fashion")  
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

xtest,xother,ttest,tother = utils.trainTestSpilt(xtest,ttest,testRate)

xtrain,ttrain = utils.multiClassTrain(xtrain,ttrain)

xtest,ttest = utils.multiClassTest(xtest,ttest)

ttrain = ttrain.astype('int')

ttest = ttest.astype('int')

svc.fit(xtrain, ttrain)

y_pred = svc.predict(xtest)

errorSVM = len(np.where((y_pred == ttest ) == False)[0]) 

errorRateSVM = errorSVM / ttest.shape[0]

