# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from PIL import Image
import random 
import matplotlib.pyplot as plt

npzfile=np.load(r'D:\Users\user\Desktop\碩班課程\金融科技\類神經\CBCL.npz')
trainface=npzfile['arr_0']
trainnonface=npzfile['arr_1']
testface=npzfile['arr_2']
testnonface=npzfile['arr_3']

I=Image.fromarray(trainface[0,:].reshape((19,19)))
I.show()

raw=np.zeros((40*19,50*19))
for y in range(40):
    for x in range(50):
        I1=trainface[y*50+x,:].reshape((19,19))
        raw[y*19:y*19+19,x*19:x*19+19]=I1
        
I=Image.fromarray(raw)
I.show()

#batchsize：train n 筆資料後一起更新(gradient),會較有效率,有的資料往右更新有的往左更新可以抵銷
def BPNNtrain(pf,nf,hn,lr,iteration): #pf：所有正資料(1),nf：所有負資料(0),hn(hidden node)：隱藏層有幾個節點(此例只有1個隱藏層),lr(learning rate)
    pn=pf.shape[0] #正資料筆數
    nn=nf.shape[0] #負資料筆數
    fn=pf.shape[1] #特徵數
    feature=np.append(pf,nf,axis=0) #正、負資料合併
    target=np.append(np.ones((pn,1)),np.zeros((nn,1)),axis=0) #產生target,正資料1、負資料0
    #隨機給初始的權重
    WI=np.random.normal(0,1,(fn+1,hn)) #weight input,fn個特徵+1個常數向項
    WO=np.random.normal(0,1,(hn+1,1))  #weight output
    for t in range(iteration):
        print(t)
        s=random.sample(range(pn+nn),pn+nn) #把資料打散
        for i in range(pn+nn):
            ins=np.append(feature[s[i],:],1)
            oh=ins.dot(WI)  #input*weight
            oh=1/(1+np.exp(-oh)) #sigmoid function
            hs=np.append(oh,1)
            out=hs.dot(WO)
            out=1/(1+np.exp(-out)) #最後的output
            #更新
            dk=out*(1-out)*(target[s[i]]-out)   #delta k(ouput node要被更新的量)
            dh=oh*(1-oh)*WO[:hn,0]*dk       #delta h(hidden node要被更新的量)
            WO[:,0]+=lr*dk*hs
            for j in range(hn): #更新每一個input到hidden的weight
                WI[:,j]+=lr*dh[j]*ins
    model=dict()
    model['WI']=WI
    model['WO']=WO
    return model

def BPNNtest(feature,model):
    sn=feature.shape[0]
    WI=model['WI']
    WO=model['WO']
    hn=WI.shape[1]
    out=np.zeros((sn,1))
    for i in range(sn):
        ins=np.append(feature[i,:],1)
        oh=ins.dot(WI)
        oh=1/(1+np.exp(-oh))
        hs=np.append(oh,1)
        out[i]=hs.dot(WO)
        out[i]=1/(1+np.exp(-out[i]))
    return out

network=BPNNtrain(trainface/255,trainnonface/255,20,0.01,10) #input資料需要做minmax到0~1,所以/255
pscore=BPNNtest(trainface/255,network)
nscore=BPNNtest(trainnonface/255,network)

X=np.zeros(99)
Y=np.zeros(99)

for i in range(99):
    threshold=(i+1)/100
    X[i]=np.mean(nscore>threshold)
    Y[i]=np.mean(pscore>threshold)   
    
plt.plot(X,Y)
    


network=BPNNtrain(testface/255,testnonface/255,20,0.01,10)
pscore=BPNNtest(testface/255,network)
nscore=BPNNtest(testnonface/255,network)
X=np.zeros(99)
Y=np.zeros(99)

for i in range(99):
    threshold=(i+1)/100
    X[i]=np.mean(nscore>threshold)
    Y[i]=np.mean(pscore>threshold)   
    
plt.plot(X,Y)
