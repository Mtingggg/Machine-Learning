# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:13:48 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
from collections import Counter
from sklearn import datasets
iris = datasets.load_iris()

#Kmeans (unsupervise algorithm)
def kmeans(sample,K,maxiter):
    N=sample.shape[0]
    D=sample.shape[1]
    C=np.zeros((K,D))   #儲存中心點的座標
    L=np.zeros((N,1))   
    L1=np.zeros((N,1))
    dist=np.zeros((N,K))
    idx=random.sample(range(N),K)
    C=sample[idx,:] #隨機取三個點當作中心點
    iter=0
    while(iter<maxiter):
        for i in range(K):
            dist[:,i]=np.sum((sample-np.tile(C[i,:],(N,1)))**2,1)  #C[i,:]原本是1X2 matrix，tile N讓他重複變成NX2 matrix(row方向重複N遍)
        L1=np.argmin(dist,1) #argmin回傳的是index,但是一個list
        if(iter>0 and np.array_equal(L,L1)): #這次label跟上一次一模一樣就break
            break
        L=L1
        for i in range(K):
            idx=np.nonzero(L==i)[0] #取出所有L(label)==i時的座標點，在下面算出這些點的新的群中心
            if(len(idx)>0):
                C[i,:]=np.mean(sample[idx,:],0)
        iter+=1
    wicd=np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1))) #sample-C[L,:]表示將所有sample減掉被分到類別的中心點
                                                       #wicd(with-in class differnce)表示所有樣本距離自己群中心的距離
    return C,L,wicd

#原始資料
C1,L1,wicd1=kmeans(iris.data,3,10000)
print(wicd1)

#standardize
stdiris=(iris.data-np.tile(np.mean(iris.data,0),(iris.data.shape[0],1)))/np.tile(np.std(iris.data,0),(iris.data.shape[0],1))
C2,L2,wicd2=kmeans(iris.data,3,10000)
wicd2=np.sum(np.sqrt(np.sum((L2-iris.target)**2)))
print(wicd2)

#min-max
minmaxiris=(iris.data-np.tile(np.min(iris.data,0),(iris.data.shape[0],1)))/(np.tile((np.max(iris.data,0)-np.min(iris.data,0)),(iris.data.shape[0],1)))
C3,L3,wicd3=kmeans(minmaxiris,3,10000)
wicd3=np.sum(np.sqrt(np.sum((L3-iris.target)**2)))
print(wicd3)



def knn(test,train,target,k): #test是要分類的資料，train是字典，target是train的label
    N=train.shape[0]
    dist=np.sum((np.tile(test,(N,1))-train)**2,1)
    idx=sorted(range(len(dist)),key=lambda i :dist[i])[0:k]  #key=lambda i :dist[i]代表用dist[i]來排序，但回傳的值是i
    return Counter(target[idx]).most_common(1)[0][0]  #Counter類似value_counts,配合counter用的

targets=iris.target[range(150)]
for k in range(1,11):
    confusionM=pd.DataFrame(0, index=np.arange(3), columns=["0","1","2"])
    finaltarget=[]
    for i in range(150):
        idx=list(range(150))
        test=iris.data[idx.pop(i)]
        train=iris.data[idx]
        target=iris.target[idx]
        pred=knn(test,train,target,k)
        finaltarget.append(pred)
    finaltarget=np.array(finaltarget).reshape(1,150).flatten()
    for a in range(3):
        index=np.where(targets==a)
        pred=finaltarget[index]        
        for b in range(3):
            confusionM.iloc[a,b]=sum(pred==b)
    print(" ")
    print("k =",k)
    print(confusionM)
