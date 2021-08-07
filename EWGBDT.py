# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:45:18 2021

@author: guwen
"""


import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import utils 
import h5py

from sklearn.metrics import f1_score

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                 max_depth=1, random_state=0)


save_dir = 'E:\AAAI2022\experiment\h5'


trainRate = 0.7

algorithm = "EWGBDT"

save_dir = 'E:\AAAI2022\experiment\h5'

data_list = ['mushroom','usps','shuttle','house','spambase','mnist','banknote','fashion']
#data_list = ['shuttle']

priors = [0.3,0.5,0.7]

lambda_dir = {'mushroom': 0.1, 'usps': 0.1, 'shuttle':0.0015,'house':0.01,
        'spambase':0.02,'mnist':0.2,'banknote':0.001,'fashion':0.1}

for dataset in data_list: 
    
    if dataset == 'banknote':
        n_unl = 400
        n_pos = 200
    else:
        n_unl = 800
        n_pos = 400
        
    print(dataset)

    lambd = lambda_dir[dataset]
    
    x,t = utils.load_dataset(dataset)   
    
    x = utils.dataPreProcess(x)
    
    xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)
    
    xtest,ttest = utils.multiClassTest(xtest,ttest)
    
    result = []
    
    for prior in priors:
        
        print(prior)
        
        f1 = []
        
        for i in range(5):
    
            P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)
            
            M = utils.computeCostMatrix(P,U)
            
            Gs = utils.sinkhornTransport(n_unl,n_pos,lambd,M)
            
            entro = utils.computeEntropy(Gs)
            
            X_train = np.r_[P, U]
            y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
            
            weight = np.r_[np.ones(P.shape[0]),entro]
            clf.fit(X_train, y_train, sample_weight=weight)
            y_pred = clf.predict(xtest)
            
            errorE = len(np.where((y_pred == ttest ) == False)[0]) 
            errorRateE = errorE / ttest.shape[0]
            
            f1EWSVM = f1_score(ttest,y_pred,average = 'micro')
            f1.append(f1EWSVM)
            
        result.append(f1)
    
    f = h5py.File(os.path.join(save_dir, algorithm + "-"  + dataset  + ".h5"), "w")
        
    f.create_dataset("f1", data=result, compression="gzip", compression_opts=9)
        
    f.close()

