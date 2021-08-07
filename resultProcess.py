# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 20:35:55 2021

@author: guwen
"""

import os
import numpy as np
import h5py
current_dir = 'E:\AAAI2022\experiment'

save_dir = 'E:\AAAI2022\experiment\h5'

data_list = ['mushroom','usps','shuttle','house','spambase','mnist','banknote','fashion']

algorithms = ['SVM','WLR','EN','PW','EWLR','EWSVM','EWGBDT']

with open(os.path.join(current_dir, "results.txt"), "w", encoding="utf-8") as write_file: # 打开需要写入的文本文件
    write_file.write('\\begin{table}[!ht]\n')
    write_file.write('\\begin{tabular}{ccccccccc}\n')
    write_file.write(" & prior & SVM & WLR & EN & PW & EWLR & EWSVM & EWGBDT \\\\ \n")
    write_file.write('\\hline \n')
    for dataset in data_list: # 遍历数据集列表
        
        performance = [] 
        avg3, std3  = [], []
        avg5, std5  = [], []
        avg7, std7  = [], []

        for algorithm in algorithms: # 遍历所有方法
            res_dir = os.path.join(current_dir,'result/' ,algorithm)
            
            pre_path = h5py.File(os.path.join(save_dir, algorithm 
                                                  + "-"  + dataset  + ".h5"), "r") # 读取前一个程序保存的文件
            measure = 'f1'
            p = pre_path[measure][:]
            
            prior_3 = p[0]
            prior_5 = p[1]
            prior_7 = p[2]
            
            performance.append(p)
            avg3.append(np.mean(prior_3)) # 计算均值
            avg5.append(np.mean(prior_5))
            avg7.append(np.mean(prior_7))
            
            std3.append(np.std(prior_3)) # 计算标准差
            std5.append(np.std(prior_5)) 
            std7.append(np.std(prior_7)) 
            
            best3 = max(avg3) # 最好的结果
            best5 = max(avg5)
            best7 = max(avg7)
        
        num = len(performance)
        
        index = 0
        
        for k in range(3):
            
            if index == 0:   
                write_file.write('  & 0.3 ')
                index += 1
            elif index == 1:
                write_file.write(' %s & 0.5 ' % dataset)
                index += 1
            elif index == 2:
                write_file.write('  & 0.7 ')
                index += 1
            
            for i in range(num):
                
                if index == 1:       
                    a, s = avg3[i], std3[i]
                    if(a == best3):
                        write_file.write(' & \\textbf{%.3f$\pm$%.3f} ' % (a, s))
                    else:
                        write_file.write(' & %.3f$\pm$%.3f ' % (a, s))
                elif index == 2:  
                    a, s = avg5[i], std5[i]
                    if(a == best5):
                        write_file.write(' & \\textbf{%.3f$\pm$%.3f} ' % (a, s))
                    else:
                        write_file.write(' & %.3f$\pm$%.3f ' % (a, s))
                elif index == 3:      
                    a, s = avg7[i], std7[i]
                    if(a == best7):
                        write_file.write(' & \\textbf{%.3f$\pm$%.3f} ' % (a, s))
                    else:
                        write_file.write(' & %.3f$\pm$%.3f ' % (a, s))
                            
                
                 
            write_file.write('\\\\ \n')
        write_file.write('\\hline \n')
        
    #write_file.write('\\\\ \n')
    write_file.write(" \\bottomrule\n")
    write_file.write(' \end{tabular}\n')
    write_file.write('\end{table}\n')