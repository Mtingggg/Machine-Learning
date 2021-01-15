# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:29:49 2019

@author: danie
"""
import math
import numpy as np
import pandas as pd

def entropy(p1,n1):
    if(p1==0 and n1==0):
        return 1
    elif(p1==0):
        return 0
    elif(n1==0):
        return 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    return -pp*math.log2(pp)-pn*math.log2(pn)

def IG(p1,n1,p2,n2):   #imformation gain
    num1 = p1+n1
    num2 = p2+n2
    num = num1+num2
    return entropy(p1+p2,n1+n2)-(num1/num*entropy(p1,n1)+num2/num*entropy(p2,n2))

import numpy as np
from sklearn import datasets
import random
iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
features = iris.data[idx,:]
targets = iris.target[idx]

def cvindex(data,nfold):
    row = list(range(data.shape[0]))
    random.shuffle(row)
    n=int(len(row)/nfold)
    return [row[i:i+n] for i in range(0, len(row), n)]

index=cvindex(features,5)
finaltarget=[]
for b in range(5):
    tempindex=index.copy()
    testindex=tempindex.pop(b)
    trainindex=tempindex
    testtarget=targets[testindex]
    testfeature=features[testindex,:]
    traintarget=targets[trainindex[0]+trainindex[1]+trainindex[2]+trainindex[3]]
    trainfeature=features[trainindex[0]+trainindex[1]+trainindex[2]+trainindex[3],:]
    for a in range(3):
        place=np.where(traintarget!=a)
        feature=trainfeature[place]
        target=traintarget[place]
        
        #decision tree
        node = dict()
        node['data'] = range(len(target))
        Tree = [];
        Tree.append(node)
        t = 0
        #leaf 1/0告訴你是否為葉節點
        #decision 不是葉節點的人會有decision,決定output為1 or 0
        #selectf 決定要用哪個feature來切
        if(a==0): #1,2
            while(t<len(Tree)):
                idx = Tree[t]['data'] #第一次迴圈時，idx=reange(0,14),第二次因為第一次有分支，所以Tree[1]['data']表示第一個分支的資料集
                #全部都是1的case
                if(sum(target[idx])==len(target[idx])):
                    Tree[t]['leaf']=1
                    Tree[t]['decision']=1
                #全部都是2的case
                elif(sum(target[idx])==len(target[idx])*2):
                    Tree[t]['leaf']=1
                    Tree[t]['decision']=2
                else:
                    bestIG = 0
                    for i in range(feature.shape[1]): #i是col數
                        pool = list(set(feature[idx,i])) #set就是unique,要找出這個資料集中，這個feature有幾種類別
                        pool.sort()
                        for j in range(len(pool)-1):     #len(pool)-1 代表有這麼多種切法
                            thres = (pool[j]+pool[j+1])/2  #找到要怎麼分支的點(這邊考慮二元分枝)
                            G1 = []
                            G2 = []
                            for k in idx:
                                if(feature[k,i]<=thres):
                                    G1.append(k)
                                else:
                                    G2.append(k)
                            thisIG = IG(sum(target[G1]==1),sum(target[G1]==2),sum(target[G2]==1),sum(target[G2]==2))
                            if(thisIG>bestIG): 
                                bestIG = thisIG
                                bestG1 = G1
                                bestG2 = G2
                                bestthres = thres
                                bestf = i
                    if(bestIG>0):
                        Tree[t]['leaf']=0
                        Tree[t]['selectf']=bestf
                        Tree[t]['threshold']=bestthres
                        Tree[t]['child']=[len(Tree),len(Tree)+1]
                        #bestIG>0，表示可以再切，切了之後樹會長左右兩邊
                        #長左邊
                        node = dict()
                        node['data'] = bestG1
                        Tree.append(node)
                        #長右邊
                        node = dict()
                        node['data'] = bestG2
                        Tree.append(node)
                    else:
                        Tree[t]['leaf']=1
                        if(sum(target[idx]==1)>sum(target[idx]==2)):
                            Tree[t]['decision']=1
                        else:
                            Tree[t]['decision']=2
                t+=1
            first=Tree.copy()
        
        elif(a==1): #0,2
            while(t<len(Tree)):
                idx = Tree[t]['data']
                #全部都是0的case
                if(sum(target[idx])==0):
                    Tree[t]['leaf']=1
                    Tree[t]['decision']=0
                #全部都是2的case
                elif(sum(target[idx])==len(target[idx])*2):
                    Tree[t]['leaf']=1
                    Tree[t]['decision']=2
                else:
                    bestIG = 0
                    for i in range(feature.shape[1]): #i是col數
                        pool = list(set(feature[idx,i])) #set就是unique,要找出這個資料集中，這個feature有幾種類別
                        pool.sort()
                        for j in range(len(pool)-1):     #len(pool)-1 代表有這麼多種切法
                            thres = (pool[j]+pool[j+1])/2  #找到要怎麼分支的點(這邊考慮二元分枝)
                            G1 = []
                            G2 = []
                            for k in idx:
                                if(feature[k,i]<=thres):
                                    G1.append(k)
                                else:
                                    G2.append(k)
                            thisIG = IG(sum(target[G1]==0),sum(target[G1]==2),sum(target[G2]==0),sum(target[G2]==2))
                            if(thisIG>bestIG): 
                                bestIG = thisIG
                                bestG1 = G1
                                bestG2 = G2
                                bestthres = thres
                                bestf = i
                    if(bestIG>0):
                        Tree[t]['leaf']=0
                        Tree[t]['selectf']=bestf
                        Tree[t]['threshold']=bestthres
                        Tree[t]['child']=[len(Tree),len(Tree)+1]
                        #bestIG>0，表示可以再切，切了之後樹會長左右兩邊
                        #長左邊
                        node = dict()
                        node['data'] = bestG1
                        Tree.append(node)
                        #長右邊
                        node = dict()
                        node['data'] = bestG2
                        Tree.append(node)
                    else:
                        Tree[t]['leaf']=1
                        if(sum(target[idx]==2)>sum(target[idx]==0)):
                            Tree[t]['decision']=2
                        else:
                            Tree[t]['decision']=0
                t+=1
            second=Tree.copy()
        else:
            while(t<len(Tree)):
                idx = Tree[t]['data'] #第一次迴圈時，idx=reange(0,14),第二次因為第一次有分支，所以Tree[1]['data']表示第一個分支的資料集
                #全部都是0的case
                if(sum(target[idx])==0):
                    Tree[t]['leaf']=1
                    Tree[t]['decision']=0
                #全部都是1的case
                elif(sum(target[idx])==len(idx)):
                    Tree[t]['leaf']=1
                    Tree[t]['decision']=1
                else:
                    bestIG = 0
                    for i in range(feature.shape[1]): #i是col數
                        pool = list(set(feature[idx,i])) #set就是unique,要找出這個資料集中，這個feature有幾種類別
                        pool.sort()
                        for j in range(len(pool)-1):     #len(pool)-1 代表有這麼多種切法
                            thres = (pool[j]+pool[j+1])/2  #找到要怎麼分支的點(這邊考慮二元分枝)
                            G1 = []
                            G2 = []
                            for k in idx:
                                if(feature[k,i]<=thres):
                                    G1.append(k)
                                else:
                                    G2.append(k)
                            thisIG = IG(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                            if(thisIG>bestIG): 
                                bestIG = thisIG
                                bestG1 = G1
                                bestG2 = G2
                                bestthres = thres
                                bestf = i
                    if(bestIG>0):
                        Tree[t]['leaf']=0
                        Tree[t]['selectf']=bestf
                        Tree[t]['threshold']=bestthres
                        Tree[t]['child']=[len(Tree),len(Tree)+1]
                        #bestIG>0，表示可以再切，切了之後樹會長左右兩邊
                        #長左邊
                        node = dict()
                        node['data'] = bestG1
                        Tree.append(node)
                        #長右邊
                        node = dict()
                        node['data'] = bestG2
                        Tree.append(node)
                    else:
                        Tree[t]['leaf']=1
                        if(sum(target[idx]==1)>sum(target[idx]==0)):
                            Tree[t]['decision']=1
                        else:
                            Tree[t]['decision']=0
                t+=1
            third=Tree.copy()

    for k in range(3):
        if k==0:
            predfirst=[]
            Tree=first
            for l in range(len(testtarget)):
                #逐行跑決策樹
                test_feature=testfeature[l,:]
                now = 0
                while(Tree[now]['leaf']==0):
                    #now 表示現在在第幾棵樹，依照規則篩選後，到下一棵樹，直到該棵樹leaf==1，表示到了葉節點
                    if(test_feature[Tree[now]['selectf']]<=Tree[now]['threshold']):
                        now = Tree[now]['child'][0]
                    else:
                        now = Tree[now]['child'][1]
                predfirst.append(Tree[now]['decision'])
                
        elif k==1:
            predsecond=[]
            Tree=second
            for l in range(len(testtarget)):
                #逐行跑決策樹
                test_feature=testfeature[l,:]
                now = 0
                while(Tree[now]['leaf']==0):
                    #now 表示現在在第幾棵樹，依照規則篩選後，到下一棵樹，直到該棵樹leaf==1，表示到了葉節點
                    if(test_feature[Tree[now]['selectf']]<=Tree[now]['threshold']):
                        now = Tree[now]['child'][0]
                    else:
                        now = Tree[now]['child'][1]
                predsecond.append(Tree[now]['decision'])                
        
        else:
            predthird=[]
            Tree=third
            for l in range(len(testtarget)):
                #逐行跑決策樹
                test_feature=testfeature[l,:]
                now = 0
                while(Tree[now]['leaf']==0):
                    #now 表示現在在第幾棵樹，依照規則篩選後，到下一棵樹，直到該棵樹leaf==1，表示到了葉節點
                    if(test_feature[Tree[now]['selectf']]<=Tree[now]['threshold']):
                        now = Tree[now]['child'][0]
                    else:
                        now = Tree[now]['child'][1]
                predthird.append(Tree[now]['decision'])
    final=np.zeros(len(predfirst))
    for p in range(len(predfirst)):
        if(sum([predfirst[p]==0,predsecond[p]==0,predthird[p]==0])>=2):
            final[p]=0
        elif(sum([predfirst[p]==1,predsecond[p]==1,predthird[p]==1])>=2):
            final[p]=1
        elif(sum([predfirst[p]==2,predsecond[p]==2,predthird[p]==2])>=2):
            final[p]=2
        else:
            final[p]=1
            
    finaltarget.append(final)            

#準確率
#total=0
#for i in range(5):
#    total+=sum(targets[index[i]]==finaltarget[i])
#print("precision:",total/150)           


finaltarget=np.array(finaltarget).reshape(1,150).flatten()
index=np.array(index).reshape(1,150).flatten()
targets=targets[index]
print(sum(targets==finaltarget)/150)

confusionM=pd.DataFrame(0, index=np.arange(3), columns=["0","1","2"])

for i in range(3):
    index=np.where(targets==i)
    pred=finaltarget[index]        
    for j in range(3):
        confusionM.iloc[j,i]=sum(pred==j)
print(confusionM.T)