# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 00:22:22 2021

@author: guwen
"""

import numpy as np
from sklearn.svm import SVC
import utils 
import codecs

from sklearn.metrics import f1_score

import matplotlib.pylab as pl

from matplotlib import pyplot as plt

C = 10

svc = SVC(C, kernel='rbf', gamma='auto', probability=True, random_state=2018)


n_unl = 800

n_pos = 400

prior = 0.7

trainRate = 0.7

algorithm = "Entropy Weighted SVM"

dataset = "mushroom"


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

lambda_set = [0.005,0.01,0.05,0.1,0.2,0.5,1]

priors = [0.3,0.5,0.7]

class_prior_list = []


x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

for prior in priors:
    P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)    
    M = utils.computeCostMatrix(P,U)
    f1_list = []
    for lambd in lambda_set:
        
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
        f1_list.append(f1E)
    class_prior_list.append(f1_list)
    

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xticks(lambda_set)
ax.set_xticklabels(['0.005','0.01','0.05','0.1','0.2','0.5','1'])

ax.set_xscale("log")
ax.set_xlabel('$\lambda$')
ax.set_ylabel('F1 Score')

l1, = ax.plot(lambda_set, class_prior_list[0], marker='o', linestyle='--')
l2, = ax.plot(lambda_set, class_prior_list[1], marker='o', linestyle='-.')
l3, = ax.plot(lambda_set, class_prior_list[2], marker='o', linestyle='-.')


ax.legend(handles=[l1, l2, l3], labels=['$\pi$ = 0.3.', '$\pi$ = 0.5',
  '$\pi$ = 0.7'], loc='lower left',bbox_to_anchor=(0,0.45))
fig.savefig(dataset+'_parameter_analysis.pdf')

"""    
ticks = [i for i in range(len(lambda_set))]

plt.xscale("log")
plt.xlabel('class prior')
plt.ylabel('F1 Score')
    
plt.xticks(ticks, ['0.005','0.01','0.05','0.1','0.2','0.5','1'],rotation='vertical')
l1, = plt.plot(lambda_set, class_prior_list[0], marker='o', linestyle='--')
l2, = plt.plot(lambda_set, class_prior_list[1], marker='o', linestyle='-.')
l3, = plt.plot(lambda_set, class_prior_list[2], marker='o', linestyle='-.')


plt.legend(handles=[l1, l2, l3], labels=['$\lambda$ = 0.3.', '$\lambda$ = 0.5',
  '$\lambda$ = 0.7'], loc='lower left',bbox_to_anchor=(0,0.45))
"""