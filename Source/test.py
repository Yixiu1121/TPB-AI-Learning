from cmath import isnan
import pandas as pd
import numpy as np
import math 
#各欄位的類型

#讀取資料並且同時設置每個欄位的類型
# T = pd.read_csv("TPB-AI-Learning\\Source\\2021_1102_tide.csv", header = None)
# T.fillna(value='Not Found', inplace=True)
# Danshui202106 = []

# for i in range(len(T.iloc[:,0])):
#     if str(T.iloc[i,0]) > "2021060100":
#         Danshui202106.append((T.iloc[i,:2]))
#         if  T.iloc[i,1]=='Not Found':
#             ## 內插 a*m+b*n)/(m+n)
#             n = 1
#             m = 1
#             while n<10:
#                 if  T.iloc[i,1+n]!='Not Found':
#                     a = int(T.iloc[i,1+n])
#                     print(a)
#                     break
#                 else:
#                     n+=1
#             while n<10:
#                 if T.iloc[i-1,-m]!='Not Found':
#                     b = int(T.iloc[i-1,-m])
#                     print(b)
#                     break
#                 else:
#                     m+=1
#             ans = str((a*m+b*n)/(m+n))
#             print(T.iloc[i,0],ans)
#             T.iloc[i,1]=ans
#             Danshui202106.append(T.iloc[i,:2])
# pd.DataFrame(Danshui202106).to_csv("Danshui202106(1).csv")

##4734

TT = pd.read_csv("TPB-AI-Learning\\Source\\2022_1102_tide.csv",header = None)
TT.fillna(value='Not Found', inplace=True)
Danshui2022 = []
for i in range(1,len(TT.iloc[:,0])):
    if  TT.iloc[i,1]=='Not Found':
        ## 內插 a*m+b*n)/(m+n)
        n = 1
        m = 1
        while n<10:
            if  TT.iloc[i,1+n]!='Not Found':
                a = int(TT.iloc[i,1+n])
                print(a)
                break
            else:
                n+=1
        while n<10:
            if TT.iloc[i-1,-m]!='Not Found':
                b = int(TT.iloc[i-1,-m])
                print(b)
                break
            else:
                m+=1
        ans = str((a*m+b*n)/(m+n))
        print(TT.iloc[i,0],ans)
        TT.iloc[i,1]=ans
        Danshui2022.append(TT.iloc[i,:2])
    else:
        Danshui2022.append((TT.iloc[i,:2]))
pd.DataFrame(Danshui2022).to_csv("Danshui2022(1).csv")
