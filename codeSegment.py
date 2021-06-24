# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:08:50 2021

@author: Administrator
"""

lambd = 0.1
x = utils.dataPreProcess(x)
xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)
xtest,ttest = utils.multiClassTest(xtest,ttest)
P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)
M = utils.computeCostMatrix(P,U)
Gs = utils.sinkhornTransport(n_unl,n_pos,lambd,M)
entro = utils.computeEntropy(Gs)
X_train = np.r_[P, U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
weight = np.r_[np.ones(P.shape[0]),entro]
svc.fit(X_train, y_train, sample_weight=weight)
y_pred = svc.predict(xtest)
errorE = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateE = errorE / ttest.shape[0]