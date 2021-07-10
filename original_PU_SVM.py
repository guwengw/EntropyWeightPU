# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:04:07 2021

@author: Administrator
"""

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


n_unl = 400

n_pos = 200

prior = 0.7

trainRate = 0.7

algorithm = "SVM"

dataset = "banknote"


x,t = utils.load_dataset(dataset)  
#mushroom 1 3916 0 4208
#usps 0-1 ~0-1 0.1
#shuttle C = 10,lamda 0.0008
#house 1 8914 0 11726
#spambase 1 1813 0 2788  C=100 lamda 0.015
#mnist
#banknote 1 762 0 610
#fashion


x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)



X_train = np.r_[P, U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]


svc.fit(X_train, y_train)
y_pred = svc.predict(xtest)

errorSVM = len(np.where((y_pred == ttest ) == False)[0]) 
errorRateSVM = errorSVM / ttest.shape[0]

f1SVM = f1_score(ttest,y_pred,average = 'micro')

with codecs.open('result.txt',mode='a') as file_txt:
    file_txt.write(algorithm+'\t'+dataset+'\t'+"prior = "+str(prior)+'\t'
                   +"SVM C = "+ str(C)+'\t'+"error = " +str(errorRateSVM)+'\t'+"F_score is "+str(f1SVM)+'\n')