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
from sklearn.metrics import f1_score

np.random.seed(47)

prior = 0.7

labeled_percent = 50

pos_size = 2000

#neg_size = 250
neg_size = int(500*(1-labeled_percent/100)/(1-prior))


validation_percent = 10
test_percent = 60
pos_val_size = pos_size * validation_percent // 100
neg_val_size = neg_size * validation_percent // 100
pos_test_size = pos_size * test_percent // 100
neg_test_size = neg_size * test_percent // 100





dataset = "mushroom"


x, t = utils.load_dataset(dataset)
X_pos = x[t==1]
X_neg = x[t==0]



np.random.shuffle(X_pos)
np.random.shuffle(X_neg)

X_test_pos, X_val_pos = X_pos[: pos_test_size], X_pos[pos_test_size: pos_test_size+pos_val_size]
X_train_pos = X_pos[pos_test_size+pos_val_size: ]

X_test_neg, X_val_neg = X_neg[: neg_test_size], X_neg[neg_test_size: neg_test_size+neg_val_size]
X_train_neg = X_neg[neg_test_size+neg_val_size: ]

np.random.shuffle(X_train_pos)
np.random.shuffle(X_train_neg)


labeled_size = X_train_pos.shape[0] * labeled_percent // 100
X_labeled = X_train_pos[: labeled_size]
X_unlabeled = np.vstack([X_train_pos[labeled_size :], X_train_neg])

X_lab_size =  X_labeled.shape[0]
X_unlab_size = X_unlabeled.shape[0]

y_lab = np.ones((X_lab_size, 1))
y_unlab = np.zeros((X_unlab_size, 1))
y = np.vstack([y_lab, y_unlab])
X = np.vstack([X_labeled, X_unlabeled])
X_y = np.hstack([X, y])

np.random.shuffle(X_y)

g_x = LogisticRegression(penalty='l2',C=1.0,solver='lbfgs',max_iter=1200,random_state=2021,n_jobs=-1)
 
g_x.fit(X_y[:, :-1], X_y[:, -1])

val_size_to_approx = 30
P = X_val_pos[:val_size_to_approx]

sumP = sum(g_x.predict_proba(P)[:, 1])

e_1 = sumP / val_size_to_approx

print(f"real c = {labeled_percent}, estimated e_1 = {e_1*100:.2f} on {val_size_to_approx} objects")

prob_pos = g_x.predict_proba(X_test_pos)[:, 1]/e_1
prob_neg = g_x.predict_proba(X_test_neg)[:, 1]/e_1
#prob_pos = g_x.predict_proba(X_test_pos)[:, 1]/(labeled_percent/100)
#prob_neg = g_x.predict_proba(X_test_neg)[:, 1]/(labeled_percent/100)


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