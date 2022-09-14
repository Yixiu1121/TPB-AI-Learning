from cmath import isnan
import pandas as pd
import numpy as np
import math 
#各欄位的類型

#讀取資料並且同時設置每個欄位的類型
T = pd.read_csv("Source/2021_1102_tide.csv", header = None)
T.fillna(value='Not Found', inplace=True)
Danshui202106 = []

for i in range(len(T.iloc[:,0])-1):
    if T.iloc[i,0] > 2021010600:
        Danshui202106.append((T.iloc[i,:2]))
        if  T.iloc[i,1]=='Not Found':
            ## 內插
            n = 1
            while n<10:
                if  T.iloc[i,1+n]!='Not Found':
                    T.iloc[i,1]=str((int(T.iloc[i-1,-2])+int(T.iloc[i,1+n]))/(1+n)+int(T.iloc[i-1,-2]))
                    Danshui202106.append(T.iloc[i,:2])
                else:
                    n+=1
pd.DataFrame(Danshui202106).to_csv("Danshui202106.csv")

TT = pd.read_csv("Source/2022_1102_tide.csv",header = None)
Danshui2022 = []
for i in range(len(TT.iloc[:,0])):
    if TT.iloc[i,1] == isnan:
        TT.iloc[i,1]=(TT.iloc[i-1,-2]+TT.iloc[i,2])/2
        Danshui2022.append(TT.iloc[i,:2])
    else:
        Danshui2022.append(TT.iloc[i,:2])
pd.DataFrame(Danshui2022).to_csv("Danshui2022.csv")
