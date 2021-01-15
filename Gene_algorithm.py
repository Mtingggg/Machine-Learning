# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:22:10 2019

@author: danie
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

data=np.load(r"C:\Users\danie\OneDrive\桌面\data.npy")
trueprice=np.log(data)

def params(tc,beta,omega,phi): #前面4個params是一個值
    t=np.array(range(tc)).reshape(tc,1)
    tc=np.tile(tc,(len(t),1))
    P=trueprice[:len(t)]
    X=np.tile(1,(len(t),1))
    Y=((tc-t)**beta)
    Z=((tc-t)**beta)*(np.cos(omega*np.log(tc-t)+phi))
    A=np.concatenate((X,Y,Z),axis=1)
    return np.linalg.lstsq(A,P,rcond=None)[0]

def E(pestimate,truedata): #error
    return np.sum(abs(pestimate-truedata))

pop=np.random.randint(0,2,(10000,13)) #初代人口,一萬個人的基因
fit=np.zeros((10000,1))

for generation in range(10):
    print('generation:',generation)
    for i in range(10000):
        gene = pop[i,:]
        tc=(np.sum(2**np.array(range(4))*gene[:4])+1151)  #把2進位數字轉10進位
        t=np.array(range(tc)).reshape(tc,1)
        beta=(np.sum(2**np.array(range(2))*gene[4:6]))*0.2+0.2
        phi=(np.sum(2**np.array(range(3))*gene[6:9]))*0.8+0.5
        omega=(np.sum(2**np.array(range(4))*gene[9:]))+1
        b=params(tc,beta,omega,phi)
        A=b[0]
        B=b[1]
        C=b[2]/B
        pestimate=A+(B*(tc-t)**beta)*(1+C*np.cos(omega*np.log(tc-t)+phi))
        fit[i]=E(pestimate,trueprice)
    #適者生存
    sortf=np.argsort(fit[:,0]) #小到大排序，最適合活的放第一個，回傳的是index
    pop=pop[sortf,:]
    print('fuck')
    for i in range(100,10000):  #只有前一百名會活著，100名後會被覆蓋掉
        #交配
        fid = np.random.randint(0,100) #father
        mid = np.random.randint(0,100) #mother
        while mid==fid: #避免爸媽同一人
            mid=np.random.randint(0,100)
        mask=np.random.randint(0,2,(1,13))
        son=pop[mid,:]
        father=pop[fid,:]
        son[mask[0,:]==1]=father[mask[0,:]==1] #mask==1的地方拿爸爸的，mask==0的地方拿媽媽的
        pop[i,:]=son
    print('evolution')
    #突變
    for i in range(1000):
        m=np.random.randint(0,1000)
        n=np.random.randint(0,13)
        pop[m,n]=1-pop[m,n] #隨機1000個位子0變1，1變0


for i in range(10000):
    gene = pop[i,:]
    tc=(np.sum(2**np.array(range(4))*gene[:4])+1151)  #把2進位數字轉10進位
    t=np.array(range(tc)).reshape(tc,1)
    beta=(np.sum(2**np.array(range(2))*gene[4:6]))*0.2+0.2
    phi=(np.sum(2**np.array(range(3))*gene[6:9]))*0.8+0.5
    omega=(np.sum(2**np.array(range(4))*gene[9:]))+1
    b=params(tc,beta,omega,phi)
    A=b[0]
    B=b[1]
    C=b[2]/B    
    pestimate=A+(B*(tc-t)**beta)*(1+C*np.cos(omega*np.log(tc-t)+phi))
    fit[i]=E(pestimate,trueprice)
sortf=np.argsort(fit[:,0]) #小到大排序，最適合活的放第一個，回傳的是index
pop=pop[sortf,:]  

gene=pop[0,:]  
tc=(np.sum(2**np.array(range(4))*gene[:4])+1151)  #把2進位數字轉10進位
t=np.array(range(tc)).reshape(tc,1)
beta=(np.sum(2**np.array(range(2))*gene[4:6]))*0.2+0.2
phi=(np.sum(2**np.array(range(3))*gene[6:9]))*0.8+0.5
omega=(np.sum(2**np.array(range(4))*gene[9:]))+1
b=params(tc,beta,omega,phi)
A=b[0]
B=b[1]
C=b[2]/B    
    
bestparams=[tc,beta,phi,omega,A,B,C]   
estimateprice=A+(B*(tc-t)**beta)*(1+C*np.cos(omega*np.log(tc-t)+phi))

plt.plot(t,estimateprice,'r',t,trueprice[:len(t)],'b')
