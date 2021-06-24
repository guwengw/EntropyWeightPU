# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:15:21 2021

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
import utils 

svc = SVC(C=100, kernel='rbf', gamma='auto', probability=True, random_state=2018)


n_unl = 500

n_pos = 300

prior = 0.5

trainRate = 0.5

lambd = 0.015

x,t = utils.load_dataset("houses")  
#mushroom 1 3916 0 4208
#banknote 1 762 0 610
#optdigits 1 572 0 5048
#spambase 1 1813 0 2788  C=100 lamda 0.015

x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)

M = utils.computeCostMatrix(P,U)

Gs = utils.sinkhornTransport(n_unl,n_pos,lambd,M)

entro = utils.computeEntropy(Gs,n_unl)

X_train = np.r_[P, U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]

weight = np.r_[np.ones(P.shape[0]),entro]
svc.fit(X_train, y_train, sample_weight=weight)
y_pred = svc.predict(xtest)

errorE = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateE = errorE / ttest.shape[0]

