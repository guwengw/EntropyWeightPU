# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:11:25 2021

@author: Administrator
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import utils 
import codecs

from sklearn.metrics import f1_score

lr = LogisticRegression(penalty='l2',C=1.0,solver='lbfgs',max_iter=1200,random_state=2021,n_jobs=-1)


n_unl = 800

n_pos = 400

prior = 0.7

trainRate = 0.7

algorithm = "Entropy Weighted Logistic Regression"

dataset = "fashion"

x,t = utils.load_dataset(dataset)  
#shuttle ~1 = 0 0.001 0.0007更好
#mushroom 1 3916 0 4208  0.1
#banknote 1 762 0 610 0.001
#optdigits 1 572 0 5048
#spambase 1 1813 0 2788  C=100 lamda 0.015
#page-blocks1 4913  0 560
#diabetes 1 268 0 500
#balance 1 288 0 337
#credit 1 700 0 300
#madelon 1 1300 0 1300
#house 1 8914 0 11726 0.005
#wdbc 1 357 0 212
#usps 0.005  0.1
#mnist

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
lr.fit(X_train, y_train, sample_weight=weight)

y_pred = lr.predict(xtest)

errorELR = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateELR = errorELR / ttest.shape[0]

f1ELR = f1_score(ttest,y_pred,average = 'micro')

with codecs.open('result.txt',mode='a') as file_txt:
    file_txt.write(algorithm+'\t'+dataset+'\t'+"prior = "+str(prior)+'\t'+"lambd is "+str(lambd)+'\t'
                   +"error = " +str(errorRateELR)+'\t'+"F_score is "+str(f1ELR)+'\n')