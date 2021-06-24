# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:58:15 2021

@author: Administrator
"""
import numpy as np
import utils 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',C=1.0,solver='lbfgs',max_iter=1200,random_state=2021,n_jobs=-1)

n_unl = 800

n_pos = 400

prior = 0.5

trainRate = 0.7


x,t = utils.load_dataset("shuttle")  #1 3916 0 4208

x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)

X_train = np.r_[P,U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]

pos_weight = len(U)/len(X_train)
neg_weight = 1 - pos_weight
assert pos_weight > neg_weight > 0

weight = [pos_weight if i else neg_weight for i in y_train]
lr.fit(X_train, y_train, sample_weight=weight)

y_pred = lr.predict(xtest)

errorLR = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateLR = errorLR / ttest.shape[0]