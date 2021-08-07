# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:16:46 2021

@author: guwen
"""

import utils
import os
#data_dir = os.path.join(......) # 数据集目录
# 要做实验的数据集列表
data_list = ['mushroom','usps','shuttle','house','spambase','mnist','banknote','fashion']
#data_list = ['fashion']
data_info = []
for name in data_list: # 遍历数据集列表
    x,tGlobal = utils.load_dataset(name)#读取数据集文件
    #print("here")
    m = x.shape[0]
    d = x.shape[1]
    tGlobal = tGlobal[tGlobal==1]
    P = tGlobal.shape[0]
    N = m - P
    data_info.append([name, m, d, P,N]) #
    #将信息保存进列表data_info
    #f.close()
# 下面根据data_info将数据集信息打印成latex文件代码
#data_info = sorted(data_info, key=(lambda x: x[2])) # 根据样本数排序
#lines = math.ceil(len(data_list) / 2) # latex表格一行显示两个数据集 故
#行数 = 数据集个数 / 2
lines = len(data_list)
with open(os.path.join("data-info.txt"), "w", encoding="utf-8") as write_file: # 打开需要写入的文本文件
    # 打印latex表格的头部信息
    write_file.write("\\begin{table}[!hb]\n")
    write_file.write(" \cnentablecaption{Basic statistics of datasets involved in the experiments}\n")
    
#    write_file.write(" ID & Data set & \#Instance & \#Feature & ID & Data
#    set & \#Instance & \#Feature \\\\\n")
    write_file.write(" ID & Data set & \#Instance & \#P / #N \\\\\n")
    write_file.write(" \hline\n")
    for i in range(lines): # 遍历所有行
    # 打印两个数据集的信息
        write_file.write(" %d & \\textsf{%s} & %d & %d & %d / %d \\\\\n" % (i+1, data_info[i][0], 
                         data_info[i][1], data_info[i][2],data_info[i][3],data_info[i][4]))
    write_file.write(" \\bottomrule\n")
    write_file.write(" \end{tabular*}\n")
    write_file.write("\end{table}\n")