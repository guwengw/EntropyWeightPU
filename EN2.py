# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:57:51 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:51:30 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import utils
sns.set(style="white")
import matplotlib.pyplot as plt
import codecs
from sklearn.metrics import f1_score

np.random.seed(47)

algorithm = "EN"



dataset = "fashion"

prior = 0.3

trainRate = 0.7

valRate = 0.1

n_unl = 800

n_pos = 400

x, t = utils.load_dataset(dataset)



x = utils.dataPreProcess(x)

xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)

xVal,xtest,tVal,ttest = utils.trainTestSpilt(xtest,ttest,valRate)

xtest,ttest = utils.multiClassTest(xtest,ttest)

X_test_pos = xtest[ttest == 1]

X_test_neg = xtest[ttest == 0]

P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)

X_train = np.r_[P,U]
y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]

g_x = LogisticRegression(penalty='l2',C=1.0,solver='lbfgs',max_iter=1200,random_state=2021,n_jobs=-1)

g_x.fit(X_train, y_train)


#val_size_to_approx = 30
#P = X_val_pos[:val_size_to_approx]
P_val = xVal[tVal == 1]

sumP = sum(g_x.predict_proba(P_val)[:, 1])

e_1 = sumP / P_val.shape[0]

c = n_pos / (n_unl * prior + n_pos)

print(f"real c = {c}, estimated e_1 = {e_1} on {P_val.shape[0]} objects")



prob_pos = g_x.predict_proba(X_test_pos)[:, 1]/e_1
prob_neg = g_x.predict_proba(X_test_neg)[:, 1]/e_1

errorEN = 1-(np.sum(prob_pos >= 0.5) + np.sum(prob_neg < 0.5))/(prob_pos.shape[0] + prob_neg.shape[0])

print("Accuracy = ", (np.sum(prob_pos >= 0.5) + np.sum(prob_neg < 0.5))/(prob_pos.shape[0] + prob_neg.shape[0]))

y_pred_pos = np.zeros(prob_pos.shape[0])
y_pred_neg = np.ones(prob_neg.shape[0])

for i in range(prob_pos.shape[0]):
    if prob_pos[i] >= 0.5:
        y_pred_pos[i] = 1
    
for i in range(prob_neg.shape[0]):
    if prob_neg[i] < 0.5:
        y_pred_neg[i] = 0
        
y_pred = np.r_[y_pred_pos,y_pred_neg]
y_test = np.r_[np.ones(prob_pos.shape[0]),np.zeros(prob_neg.shape[0])]

f1EN = f1_score(y_test,y_pred,average = 'micro')

with codecs.open('result.txt',mode='a') as file_txt:
    file_txt.write(algorithm+'\t'+dataset+'\t'+"prior = "+str(prior)+'\t'+"error = "
                    +str(errorEN)+'\t'+"F_score is "+str(f1EN)+'\n')