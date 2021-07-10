# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:40:01 2021

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
import utils 
import matplotlib.pylab as pl

n_unl = 2000

n_pos = 1000

prior = 0.5

trainRate = 0.99



dataset = "mushroom"

x,t = utils.load_dataset(dataset)  
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


pl.scatter(range(len(entro)), entro, color='r')
pl.legend([dataset])

#pl.axhline(y = 0.5,xmin=0,xmax=1,color='b',linestyle="dashed")
