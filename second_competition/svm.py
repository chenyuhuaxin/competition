# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:45:46 2019

@author: Lenovo
"""


import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.svm import SVC


def load_csv(filename):
    print('begin')
    df = pd.read_csv(filename,sep=',',header=None)
    df1=df.replace("?",0)
    df2 = df1.replace(0, df1[[0,1,2,3,4,5,6,7,8,9,10,11,12]].mean(axis=0))
    #print(df2[12])
    data=df2.astype(int)
    data = np.array(data)
    #data = data.tolist()
    #print(data)
    return data


def svc(train_data,train_label,test_data):
    clf = SVC(kernel='rbf')#调参
    clf.fit(train_data, train_label)#训练
    print(clf.fit(train_data, train_label))#输出参数设置
    p = 0#正确分类的个数
    pre = []
    for i in range(len(test_data)):#循环检测测试数据分类成功的个数
        pre.append(clf.predict([test_data[i]]))
# =============================================================================
#         if pre[i] == test_label[i]:
#             p += 1
#     print('正确率：')
#     print(p / len(test_label))#输出测试集准确率
# =============================================================================
    return pre


def main():
    filename = 'train.csv'
    data= load_csv(filename)
    x, train_y  = data[:,:13], data[:,-1]
    pca = PCA(n_components=9)
    pca.fit(x)
    train_X = pca.transform(x)
    
    test_name = 'test.csv'
    test = load_csv(test_name)
    test_X = pca.transform(test)
    
    pre = svc(train_X, train_y, test_X)
 
    num = [x+1 for x in range(len(test_X))]
    pre = [int(x) for x in pre]
    d = {'id':pd.Series(num), 'y':pd.Series(pre)}
    result = pd.DataFrame(d)
    result.to_csv("D:\大三上\机器学习框架\second_competition\svm_降维.csv", index=False)
    
    
    
if __name__ == "__main__":
    main()
    