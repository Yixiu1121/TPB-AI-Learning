# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:48:55 2022

@author: yixiu
"""
# import pandas as pd

# ShihmenData = pd.read_csv(r"C:\Users\yixiu\OneDrive\文件\台北橋1\ShihmenRecord-utf8.csv",usecols=["水情時間","進流量(cms)","出流量(cms)"])
# ShihmenData.to_csv(r"C:\Users\yixiu\OneDrive\文件\台北橋1\石門.csv", index=False, header = False)

# def DataFrametoList(DF):
#     return DF.values.tolist()
# Data1 = DataFrametoList(ShihmenData)
Sec = []
Sec.append(['2021/1/1 07:00','進流量(cms)','出流量(cms)'])
Sec.append(['2021/1/1 08:00','進流量(cms)','出流量(cms)'])
Sec.append(['2021/1/1 09:00','aaaaa','bbbbb'])
Sec.append(['2021/1/1 10:00','ccc','dddd'])

RRR = []
RRR.append(['2021/1/1 06:00',1])
RRR.append(['2021/1/1 08:00',2])
RRR.append(['2021/1/1 09:00',3])
RRR.append(['2021/1/1 11:00',4])

TT = []
TT.append(['2021/1/1 06:00',111])
TT.append(['2021/1/1 08:00',200])
TT.append(['2021/1/1 09:00',300])
#TT.append(['2021/1/1 11:00',414])

class TPB():
    Date:str
    outflow:int
    inflow:int
    Tide:int
    Rainfall:int
   

class Reservoir():
    Date:str
    outflow:int
    inflow:int
    
    
class Rainfall():
    Date:str
    rainfall:int
   

class Tide():
    Date:str
    tide:int


S = Reservoir()
S.Date = [Sec[i][0] for i in range(len(Sec))] 
S.inflow =  [Sec[i][1] for i in range(len(Sec))] 
S.outflow = [Sec[i][2] for i in range(len(Sec))]

R = Rainfall()
R.Date = [RRR[i][0] for i in range(len(RRR))] 
R.rainfall = [RRR[i][1] for i in range(len(RRR))]

T = Tide()
T.Date = [TT[i][0] for i in range(len(TT))] 
T.tide = [TT[i][1] for i in range(len(TT))]  

Resultdic = TPB()
Resultdic = {}
#Resultdic1 = {Resultdic.Date:[Resultdic.outflow,Resultdict.Tide]}

paranum = 0

## 第一組變數
for i in range(len(S.Date)):
    if S.Date[i] not in Resultdic:
        Resultdic[S.Date[i]]=[S.outflow[i],S.inflow[i]]
    elif S.Date[i] in Resultdic:
        Resultdic[S.Date[i]].extend([S.outflow[i],S.inflow[i]])
paranum = paranum+2

## 第二組變數
for i in range(len(R.Date)):
    if R.Date[i] not in Resultdic:
        Resultdic[R.Date[i]]=[-999,-999,R.rainfall[i]]
    elif R.Date[i] in Resultdic:
        Resultdic[R.Date[i]].extend([R.rainfall[i]])
        
for i in Resultdic.keys():
    if len(Resultdic[i])< paranum :
        Resultdic[i].extend([-999])
paranum = paranum+1


## 第三組變數
for i in range(len(T.Date)):
    if T.Date[i] not in Resultdic:
        Resultdic[T.Date[i]]=[-999,-999,-999,T.tide[i]]
    elif T.Date[i] in Resultdic:
        Resultdic[T.Date[i]].extend([T.tide[i]])
paranum = paranum+1

for i in Resultdic.keys():
    L = len(Resultdic[i])
    if L<paranum:
        for time in range(paranum-L):
            Resultdic[i].extend([-999])

## 排序
dict(sorted(Resultdic.items()))

## Case
Event = []
Event.append(['2021/1/1 06:00','2021/3/8 13:00'])
Event.append(['2021/1/9 12:00','2021/3/10 18:00'])

class Eventime():
    strtime:str
    endtime:str

EE = Eventime()
EE.strtime =[ Event[i][0] for i in range(len(Event))]
EE.endtime =[ Event[i][1] for i in range(len(Event))]

eNum = 0
E1 = []
for i in Resultdic.keys():
    if i > EE.strtime[eNum] and i < EE.endtime[eNum]:
        E1.append([i,i.values])
        











