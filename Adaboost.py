# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

npzfile=np.load(r'D:\Users\user\Desktop\碩班課程\金融科技\Adaboost\CBCL.npz')
trainface=npzfile['arr_0']
trainnonface=npzfile['arr_1']
testface=npzfile['arr_2']
testnonface=npzfile['arr_3']

trpn=trainface.shape[0] #train positive number
trnn=trainnonface.shape[0]
tepn=testface.shape[0]  #test positive number
tenn=testnonface.shape[0]

#計算特徵個數，並儲存所有特徵座標，共有36648個特徵
fn=0
ftable=[]
for y in range(19): #起始y
    for x in range(19): #起始X
        for h in range(2,20): #height(最小高度為2)
            for w in range(2,20): #wide(最小寬度為2)
                if(y+h<=19 and x+w*2<=19):
                    fn=fn+1
                    ftable.append([0,y,x,h,w])


for y in range(19): #起始y
    for x in range(19): #起始X
        for h in range(2,20): #height
            for w in range(2,20): #wide
                if(y+h*2<=19 and x+w<=19):
                    fn=fn+1
                    ftable.append([1,y,x,h,w])

for y in range(19): #起始y
    for x in range(19): #起始X
        for h in range(2,20): #height
            for w in range(2,20): #wide
                if(y+h<=19 and x+w*3<=19):
                    fn=fn+1
                    ftable.append([2,y,x,h,w])


for y in range(19): #起始y
    for x in range(19): #起始X
        for h in range(2,20): #height
            for w in range(2,20): #wide
                if(y+h*2<=19 and x+w*2<=19):
                    fn=fn+1
                    ftable.append([3,y,x,h,w])
                    
def fe(sample,ftable,c):    #座標左上角(0,0)
    ftype=ftable[c][0]
    y=ftable[c][1]
    x=ftable[c][2]
    h=ftable[c][3]                    
    w=ftable[c][4]         
    T=np.arange(361).reshape((19,19))
    if(ftype==0):
        idx1=T[y:y+h,x:x+w].flatten()     #白色區
        idx2=T[y:y+h,x+w:x+w*2].flatten()  #黑色區
        output=np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1) #所有圖一起加減，sample中一個row表一張圖
    elif(ftype==1):
        idx1=T[y:y+h,x:x+w].flatten()  #黑色區
        idx2=T[y+h:y+h*2,x:x+w].flatten()  #白色區
        output=np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx1],axis=1) #白-黑
    elif(ftype==2):
        idx1=T[y:y+h,x:x+w].flatten()  
        idx2=T[y:y+h,x+w:x+w*2].flatten()  
        idx3=T[y:y+h,x:x+w*2:x+w*3].flatten()        
        output=np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)
    else:
        idx1=T[y:y+h,x:x+w].flatten() 
        idx2=T[y:y+h,x+w:x+w*2].flatten() 
        idx3=T[y+h:y+h*2,x:x+w].flatten() 
        idx4=T[y+h:y+h*2,x+w:x+w*2].flatten() 
        output=np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)-np.sum(sample[:,idx4],axis=1)
    return output

trpf=np.zeros((trpn,fn))    #2429X36648
trnf=np.zeros((trnn,fn))    #4548X36648

for c in range(fn): #把所有資料都取36648個特徵，fn=36648(特徵數)
    trpf[:,c]=fe(trainface,ftable,c)
    trnf[:,c]=fe(trainnonface,ftable,c)


#==========處理原資料特徵以及計算error==========
def WC(pw,nw,pf,nf):    #pw人臉資料權重 nw非人臉資料權重 (起始為equal weight)
    maxf=max(pf.max(),nf.max()) #feature的最大值
    minf=min(pf.min(),nf.min()) #feature的最小值為了要切分資料
    #將資料分成n塊
    theta=(maxf-minf)/10+minf #將feature分成10等分，會有9個切分點
    error=np.sum(pw[pf<theta])+np.sum(nw[nf>=theta]) #分錯的就把weight加起來，人臉=1，非人臉=0，所以pf<theta就是分錯的
    polarity=1
    if(error>0.5):  #若error>0.5，表示反過來做決策比較好
        polarity=0
        error=1-error
    min_theta=theta
    min_error=error
    min_polarity=polarity
    for i in range(2,10): #第一個切分點前面已經做了，這邊剩8個
        theta=(maxf-minf)*i/10+minf
        error=np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity=1
        if(error>0.5):
            polarity=0
            error=1-error
        if(error<min_error):
           min_theta=theta
           min_error=error
           min_polarity=polarity
    return min_error,min_theta,min_polarity


#==========Adaboost訓練過程function==========

pw=np.ones((trpn,1))/trpn/2 #初始權重(相等權重)
nw=np.ones((trnn,1))/trnn/2
SC = []
for t in range(200):
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0
    for i in range(1,fn):
        me,mt,mp = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(me<best_error):
            best_error = me
            best_feature = i
            best_theta = mt
            best_polarity = mp
    beta = best_error/(1-best_error)
    if(best_polarity == 1):
        pw[trpf[:,best_feature]>=best_theta]*=beta
        nw[trnf[:,best_feature]<best_theta]*=beta
    else:
        pw[trpf[:,best_feature]<best_theta]*=beta
        nw[trnf[:,best_feature]>=best_theta]*=beta
    alpha = np.log10(1/beta) #此分類器的權重
    SC.append([best_feature,best_theta,best_polarity,alpha])
    print(t)
    print(best_feature)

##訓練完的model
num=[1,3,5,20,100,200]
for j in range(6):
    trps = np.zeros((trpn,1))
    trns = np.zeros((trnn,1))
    alpha_sum = 0
    for i in range(num[j]):
        feature = SC[i][0]
        theta = SC[i][1]
        polarity = SC[i][2]
        alpha = SC[i][3]
        alpha_sum += alpha
        if(polarity==1):
            # 我現在有10個分類器，每個分類器有不同的權重alpha，最後集成的結果為每個若分類器預測結果(這邊是0 or 1)乘上alpha(各弱分類器的權重)之加總
            trps[trpf[:,feature]>=theta] += alpha #因這邊弱分類器output是0 or 1，所以分到1的樣本直接加上權重(alpha)就可以
            trns[trnf[:,feature]>=theta] += alpha 
        else:
            trps[trpf[:,feature]<theta] += alpha
            trns[trnf[:,feature]<theta] += alpha
            
    #為了畫roc curve而將資料標準化，事實上原論文的threshold是alpha_sum/2，大於=1，小於=0
    trps /= alpha_sum
    trns /= alpha_sum
    
    x = []
    y = []
    for i in range(1000):
        threshold = i/1000
        x.append(np.sum(trns>=threshold)/trnn)
        y.append(np.sum(trps>=threshold)/trpn)
    plt.plot(x,y)
    plt.show()
        

#=========================用test data 畫ROC curve=================================
tepf=np.zeros((tepn,fn))    
tenf=np.zeros((tenn,fn))          

for c in range(int(fn/2),fn): #把所有資料都取36648個特徵，fn=36648(特徵數)
    tepf[:,c]=fe(testface,ftable,c)
    tenf[:,c]=fe(testnonface,ftable,c) 

num=[1,3,5,20,100,200]
for j in range(6):
    teps = np.zeros((tepn,1))
    tens = np.zeros((tenn,1))
    alpha_sum = 0

    for i in range(num[j]):
        feature = SC[i][0]
        theta = SC[i][1]
        polarity = SC[i][2]
        alpha = SC[i][3]
        alpha_sum += alpha
        if(polarity==1):
            # 我現在有10個分類器，每個分類器有不同的權重alpha，最後集成的結果為每個若分類器預測結果(這邊是0 or 1)乘上alpha(各弱分類器的權重)之加總
            teps[tepf[:,feature]>=theta] += alpha #因這邊弱分類器output是0 or 1，所以分到1的樣本直接加上權重(alpha)就可以
            tens[tenf[:,feature]>=theta] += alpha 
        else:
            teps[tepf[:,feature]<theta] += alpha
            tens[tenf[:,feature]<theta] += alpha
            
    #為了畫roc curve而將資料標準化，事實上原論文的threshold是alpha_sum/2，大於=1，小於=0
    teps /= alpha_sum
    tens /= alpha_sum
    
    x = []
    y = []
    for i in range(1000):
        threshold = i/1000
        x.append(np.sum(tens>=threshold)/tenn)
        y.append(np.sum(teps>=threshold)/tepn)
    plt.plot(x,y)
    plt.show()
           

#=========================找一張有人臉的照片用訓練好的分類器判斷=================================
I=Image.open(r'C:\Users\danie\OneDrive\桌面\下載.jpg')
W,H=I.size
data=np.asarray(I)  
gray=(data[:,:,0]+data[:,:,1]+data[:,:,2])/3

#產生我要判斷的照片集
pics=np.zeros((8580,361))
idx=0
for r in range(65):
    for c in range(132):
        pics[idx,:]=gray[r:r+19,c:c+19].reshape((1,361))
        idx+=1
        print(idx)

#做特徵工程
picsf=np.zeros((8580,36648))
for c in range(fn): #把所有資料都取36648個特徵，fn=36648(特徵數)
    picsf[:,c]=fe(pics,ftable,c)   
    
#一百個弱分類器判斷
alpha_sum = 0
picsresult=np.zeros((8580,1))
for i in range(100):
    print(i)
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        # 我現在有10個分類器，每個分類器有不同的權重alpha，最後集成的結果為每個若分類器預測結果(這邊是0 or 1)乘上alpha(各弱分類器的權重)之加總
        picsresult[picsf[:,feature]>=theta] += alpha #因這邊弱分類器output是0 or 1，所以分到1的樣本直接加上權重(alpha)就可以
    else:
        picsresult[picsf[:,feature]<theta] += alpha            

#picsresult>alpha_sum/2，表示模型判斷為人臉
face=np.nonzero(picsresult>=alpha_sum/2)[0]

#獲得第n張照片的boundary
def getidx(num):
    rowstart=int(num/132)
    rowend=rowstart+18
    colstart=num-rowstart*132-1
    colend=colstart+18
    return [rowstart,rowend,colstart,colend]

#在我判斷為人臉的位子畫上綠色框框
datacopy=data.copy()
for z in range(len(face)):
    faceidx=getidx(face[z])
    for rgb in range(3):
        if(rgb==1):
            datacopy[faceidx[0],faceidx[2]:faceidx[3]+1,rgb]=150
            datacopy[faceidx[1],faceidx[2]:faceidx[3]+1,rgb]=150
            datacopy[faceidx[0]:faceidx[1]+1,faceidx[2],rgb]=150
            datacopy[faceidx[0]:faceidx[1]+1,faceidx[3],rgb]=150
        else:
            datacopy[faceidx[0],faceidx[2]:faceidx[3]+1,rgb]=0
            datacopy[faceidx[1],faceidx[2]:faceidx[3]+1,rgb]=0
            datacopy[faceidx[0]:faceidx[1]+1,faceidx[2],rgb]=0
            datacopy[faceidx[0]:faceidx[1]+1,faceidx[3],rgb]=0
            
datacopy=datacopy.astype('uint8')
I2=Image.fromarray(datacopy,'RGB')
I2.show()