# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:13:27 2021

@author: Administrator
"""

for i in range(xtest.shape[0]):
    for j in range(xtrain.shape[0]):
        if (xtest[i] == xtrain[j]).all():
            print("error")