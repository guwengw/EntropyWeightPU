# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:30:18 2021

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
import utils 
import codecs

from sklearn.metrics import f1_score

C = 10

svc = SVC(C, kernel='rbf', gamma='auto', probability=True, random_state=2018)


n_unl = 800

n_pos = 400

prior = 0.7

trainRate = 0.7

algorithm = "Entropy Weighted SVM"

dataset = "fashion"


x,t = utils.load_dataset(dataset)  
#usps 0-1 ~0-1 0.1
#shuttle C = 10,lamda 0.0008
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

f1E = f1_score(ttest,y_pred,average = 'micro')

with codecs.open('result.txt',mode='a') as file_txt:
    file_txt.write(algorithm+'\t'+dataset+'\t'+"prior = "+str(prior)+'\t'+"lambd is "+str(lambd)+'\t'
                   +"SVM C = "+ str(C)+'\t'+"error = " +str(errorRateE)+'\t'+"F_score is "+str(f1E)+'\n')