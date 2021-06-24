# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:54:46 2021

@author: Administrator
"""

import numpy as np
import scipy as sp
import ot
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

def load_dataset(dataset):
    if dataset == "mushroom":
        x, t = load_svmlight_file("data/mushrooms.txt")
        x = x.toarray()
        x = np.delete(x, 77, 1)  # contains only one value
        t[t == 1] = 1
        t[t == 2] = 0
        
    elif dataset == "mnist":
        x_train, t = load_svmlight_file('data/mnist.scale')
        x = x_train.toarray()
        ##t[(t == 2)] = 1 #1-4672  0-8320
        #t[(t != 1)] = 0
        t[t == 1] = 100
        t[t == 3] = 1
        
        
    elif dataset == "usps":
        x_train, t_train = load_svmlight_file('data/usps')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('data/usps.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[t == 1] = 1
        t[t == 3] = 0
    
    elif dataset == "shuttle":
        x_train, t_train = load_svmlight_file('data/shuttle.scale.txt')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('data/shuttle.scale.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        #t[t == 1] = 10
        t[t != 1] = 0
        #t[t == 5] = 0
        ##t[t == 4] = 1
        
        
        
    elif dataset == "banknote":
        opt = fetch_openml(name= 'banknote-authentication')
        x = opt.data
        t = opt.target
        t[t == '1'] = 1
        t[t == '2'] = 0
        
    elif dataset == "fashion":
        opt = fetch_openml(name= 'Fashion-MNIST')
        x = opt.data
        t = opt.target
        t[t == '1'] = 1
        t[t == '0'] = 0
    
    elif dataset == "optdigits":
        opt = fetch_openml(name= 'optdigits',version=2)
        x = opt.data
        t = opt.target
        t[t == 'P'] = 1
        t[t == 'N'] = 0
        
    elif dataset == "spambase":
        opt = fetch_openml(name= 'spambase')
        x = opt.data
        t = opt.target
        t[t == '1'] = 1
        t[t == '0'] = 0
        
    elif dataset == "page":
        opt = fetch_openml(name= 'page-blocks',version=2)
        x = opt.data
        t = opt.target
        t[t == 'P'] = 1
        t[t == 'N'] = 0
        
    elif dataset == "diabetes":
        opt = fetch_openml(name= 'diabetes',version=1)
        x = opt.data
        t = opt.target
        t[t == 'tested_positive'] = 1
        t[t == 'tested_negative'] = 0
    
    elif dataset == "balance":
        opt = fetch_openml(name= 'balance-scale',version=2)
        x = opt.data
        t = opt.target
        t[t == 'P'] = 1
        t[t == 'N'] = 0
    
    elif dataset == "credit":
        opt = fetch_openml(name= 'credit-g',version=1)
        x = opt.data
        t = opt.target
        t[t == 'good'] = 1
        t[t == 'bad'] = 0
        
    elif dataset == "madelon":
        opt = fetch_openml(name= 'madelon',version=1)
        x = opt.data
        t = opt.target
        t[t == '1'] = 1
        t[t == '2'] = 0
        
    elif dataset == "house":
        opt = fetch_openml(name= 'houses',version=2)
        x = opt.data
        t = opt.target
        t[t == 'P'] = 1
        t[t == 'N'] = 0
    
    elif dataset == "wdbc":
        opt = fetch_openml(name= 'wdbc',version=1)
        x = opt.data
        t = opt.target
        t[t == '1'] = 1
        t[t == '2'] = 0
        
    elif dataset == "isolet":
        opt = fetch_openml(name= 'isolet',version=2)
        x = opt.data
        t = opt.target
        t[t == '1'] = 1
        t[t == '2'] = 0
        
    return x,t
        
    
def dataPreProcess(x):
    div = np.max(x, axis=0) - np.min(x, axis=0)
    div[div == 0] = 1
    x = (x - np.min(x, axis=0)) / div
    return x

def trainTestSpilt(x,t,trainRate):
    trainSize = int(t.shape[0] * trainRate)
    xtrain,xtest,ttrain,ttest = train_test_split(x, t, train_size=trainSize,
                                             random_state=2021)
    return xtrain,xtest,ttrain,ttest

def multiClassTrain(xtrain,ttrain):
    xtrain_p = xtrain[ttrain == 1]
    ttrain_p = ttrain[ttrain == 1] 
    
    xtrain_n = xtrain[ttrain == 0]
    ttrain_n = ttrain[ttrain == 0] 

    xtrain_pn = np.r_[xtrain_p, xtrain_n]
    ttrain_pn = np.r_[ttrain_p, ttrain_n]
    
    return xtrain_pn,ttrain_pn

def multiClassTest(xtest,ttest):
    xtest_p = xtest[ttest == 1]
    ttest_p = ttest[ttest == 1] 
    
    xtest_n = xtest[ttest == 0]
    ttest_n = ttest[ttest == 0] 

    xtest_pn = np.r_[xtest_p, xtest_n]
    ttest_pn = np.r_[ttest_p, ttest_n]
    
    return xtest_pn,ttest_pn
    
    
def positiveUnlabeledSpilt(x,t,n_unl,prior,n_pos):
    size_u_p = int(prior * n_unl)
    size_u_n = n_unl - size_u_p

    xp_t = x[t == 1]
    tp_t = t[t == 1]    
    
    xp, xp_other, _, tp_o = train_test_split(xp_t, tp_t, train_size=n_pos,
                                             random_state=2021)

    xup, _, _, _ = train_test_split(xp_other, tp_o, train_size=size_u_p,
                                        random_state=2021)


    xn_t = x[t == 0]
    tn_t = t[t == 0]
    xun, _, _, _ = train_test_split(xn_t, tn_t, train_size=size_u_n,
                                    random_state=1)
    xu = np.concatenate([xup, xun], axis=0)
    #yu = np.concatenate((np.ones(len(xup)), np.zeros(len(xun))))

    P = xp
    U = xu
    #U = shuffle(U)
    return P,U

def computeCostMatrix(P,U):
    M = sp.spatial.distance.cdist(U, P)
    #M = sp.spatial.distance.cdist(P, U)
    return M

def sinkhornTransport(n_unl,n_pos,lambd,M):
    p = ot.unif(n_unl)
    q = ot.unif(n_pos)
    Gs = ot.sinkhorn(p, q, M, lambd, numItermax=2000,verbose=True)
    return Gs

def computeEntropy(Gs):
    #Gs = Gs.T
    entro = np.zeros(Gs.shape[0])
    for i in range(Gs.shape[0]):
        entro[i] = sp.stats.entropy(Gs[i])
    
    entro = entro/max(entro)
    return entro

