# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 02:09:59 2022

@author: yixiu
"""

import pandas as pd

# 水庫資料、雨量、潮位
# 暴雨事件
ShihmenData = pd.read_csv("ShihmenRecord.csv",usecols=["水情時間","進流量(cms)","出流量(cms)"])
# Rainfall = pd.read_csv("ShihmenRecord.csv")
# Tide = pd.read_csv("ShihmenRecord.csv")
# Event = pd.read_csv("ShihmenRecord.csv")

def DataFrametoList(DF):
    return DF.values.tolist()
    
def TimetoInt(date:str):
    "'2021/1/8 06:00' convert to 2021010806"
    YMD, hour = date.split(' ')
    Y,M,D = YMD.split('/')
    if int(M)<10:
        M = "0"+M
    if int(D)<10:
        D = "0"+D
    hr = hour[:2] 
    date = int(Y+M+D+hr)
    return date

##事件始末和時間序列比較 存入時間序列之時間
## 第二的時間序列與第一比較 存入新的時間序列
Event = []
Event.append(['2021/1/1 06:00','2021/3/8 13:00'])
Event.append(['2021/1/9 12:00','2021/3/10 18:00'])
def TimeCompare(Event,Data):
    EVENT = []
    for CaseNum in range(len(Event)):
        # "事件場次 "
        EVENT.append([f"#{CaseNum+1}",Event[CaseNum][0],Event[CaseNum][1]])
        eventSTTime = TimetoInt(Event[CaseNum][0])
        eventEndTime = TimetoInt(Event[CaseNum][1])  
        for i in range(len(Data)):
            # "存入資料"
            if TimetoInt(Data[i][0])>=eventSTTime and TimetoInt(Data[i][0])<=eventEndTime:
                EVENT.append([Data[i]])
    return EVENT


E = []
#E.append(["#1","st1","st2"])
#E.append(['2021/1/1 06:00','進流量(cms)','出流量(cms)'])
#E.append(['2021/1/1 07:00','進流量2(cms)','出流量2(cms)'])
E.append(["#2","st1","st2"])
E.append(['2021/1/1 08:00','進流量(cms)','出流量(cms)'])
E.append(['2021/1/1 10:00','進流量2(cms)','出流量2(cms)'])
#EE =pd.DataFrame(E)

## extend 會變更到原本的list

Sec = []
Sec.append(['2021/1/1 07:00','進流量(cms)','出流量(cms)'])
Sec.append(['2021/1/1 08:00','進流量(cms)','出流量(cms)'])
Sec.append(['2021/1/1 09:00','aaaaa','bbbbb'])
Sec.append(['2021/1/1 10:00','ccc','dddd'])
#SS =TimeCompare(Event,Sec)


#Sec.extend([E[0][0],E[0][1],E[0][2],'出流量2(cms)'])
# E[0].append("出流量")
# Sec.extend([E[0]])

def CompareSeries(E:list,second:list):
    Epara = 2

    for i in range(len(E)):
        if E[i][0][0]=="#":
            pass
        else:
            for j in range(len(second)):
                if TimetoInt(second[j][0]) == TimetoInt(E[i][0]):
                    E[i].extend(Sec[j][1:])
                    print(j,second[j])
                
                elif E[i+1][0][0]!="#":
                    
                    if TimetoInt(second[j][0])>TimetoInt(E[i][0]) and TimetoInt(second[j][0])<TimetoInt(E[i+1][0]):
                        #print(i,j)
                        x = [second[j][0]]
                        [x.append(-999) for i in range(Epara)]
                        x.extend(second[j][1:])
                        E.insert(i+1,x)
                        #print(E)
                        i = i+1
    return E
NN = []
NN = CompareSeries(E=E, second=Sec)
NNN = CompareSeries(E=NN, second=Sec)


#%%
def FrameConvert(Event):
    Event = pd.DataFrame(Event)
    timeseries = []
    for i in Event["sttime"]:
        timeseries.append(TimetoInt(i))
    return timeseries
FrameConvert(Event = Event)