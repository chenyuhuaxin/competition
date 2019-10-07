# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:27:25 2019

@author: Lenovo
"""
import numpy as np
import operator
#import os
#from pandas import Series,DataFrame
import pandas as pd

def read_data():
    data = (np.loadtxt('HTRU_2_train.csv',delimiter=","))
    test_data = (np.loadtxt('HTRU_2_test.csv',delimiter=","))
    
    train_data = list()
    train_label = list()
    
    train_data = data[:,:2]
    train_label = data[:,-1]
    #print(train_data.shape)
    #print(test_label.shape)
    return train_data, train_label, test_data
    
    
def knn(trainData, testData, labels, k):
    # 计算训练样本的行数
    row = trainData.shape[0]
    # 计算训练样本和测试样本的差值
    diff = np.tile(testData, (row, 1)) - trainData
    # 计算差值平方和
    sqrDiff = diff ** 2
    sqrDiffSum = sqrDiff.sum(axis = 1)
    # 计算距离
    distances = sqrDiffSum ** 0.5
    # 对距离进行从高到低的排序
    sortDistance = distances.argsort()
    count = {}
    
    for i in range(k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    
    # 对类别出现的频率从高到低进行排序
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现频率最高的类别
    return sortCount[0][0]

def main():
    train_data, train_label, test_data= read_data()
    pre = []
    #real = 0
    for i in test_data:
        a = knn(train_data, i, train_label,35)
        pre.append(a)
    
    num = [x+1 for x in range(700)]
    pre = [int(x) for x in pre]
    d = {'id':pd.Series(num), 'y':pd.Series(pre)}
    result = pd.DataFrame(d)
    result.to_csv("D:\大三上\机器学习框架\knn35.csv", index=False)
        
    
if __name__ == '__main__':
    main()