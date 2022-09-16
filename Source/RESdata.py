## RES 整理
## 水庫出入流
print("123")
import pandas as pd
# import openpyxl
import tensorflow 

R = pd.read_csv("C:\\Users\\309\\Documents\\GitHub\\TPB-AI-Learning\\Source\\RES-SM.csv",usecols=['STA_NO','INFO_DATE','MM','Q_IN','Q_OUT'])
F = pd.read_csv("C:\\Users\\309\\Documents\\GitHub\\TPB-AI-Learning\\Source\\RES-FT.csv",usecols=['STA_NO','INFO_DATE','MM','Q_IN','Q_OUT'])
SMR = [R.iloc[0,0]]
for i in range(len(R)):   
    if R.iloc[i,2]=='0':
        SMR.append(R.iloc[i,[0,1,3,4]])

FTR = [F.iloc[0,0]]
for i in range(len(F)):   
    if F.iloc[i,2]=='0':
        FTR.append(F.iloc[i,[0,1,3,4]])


