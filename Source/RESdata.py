## RES 整理
## 水庫出入流
# print("123")
import pandas as pd
# import openpyxl
# import tensorflow 

R = pd.read_csv("TPB-AI-Learning\\Source\\RES-SM.csv",usecols=['STA_NO','INFO_DATE','MM','Q_IN','Q_OUT'],low_memory=False)
F = pd.read_csv("TPB-AI-Learning\\Source\\RES-FT.csv",usecols=['STA_NO','INFO_DATE','MM','Q_IN','Q_OUT'],low_memory=False)

def TimetoInt(date:str):
    "2022/9/15 3:50 PM convert to 202101080600"
    x = date.split(' ')
    YMD, hour = x[0],x[1]
    hr, min = hour.split(':')
    YMD = YMD.split('/')
    Y,M,D = YMD[0], YMD[1], YMD[2]
    if int(M)<10:
        M = "0"+M
    if int(D)<10:
        D = "0"+D
    if int(hr)<10:
        hr = "0"+hr
    date = int(Y+M+D+hr)
    return date
print(TimetoInt('2021/11/6 00:00'))

SMR = []
#['STA_NO','INFO_DATE','MM','Q_IN','Q_OUT']
# for i in range(1,len(R)):  
#     if  TimetoInt(R.iloc[i,1]) > 2021060100:
#         if int(R.iloc[i,2]) == 0:
#             SMR.append(R.iloc[i,1:5])
#     else:
#         print(i)
#         break
for i in range(1,len(R)):
    if int(R.iloc[i,2])==0:
        if TimetoInt(R.iloc[i,1]) > 2021060100:
            SMR.append(R.iloc[i,1:5])
        if TimetoInt(R.iloc[i,1]) < 2021060100:
            break
print('SMR')
pd.DataFrame(SMR).to_csv("SMR.csv",index=None)
FTR = []
for i in range(1,len(F)):   
    if int(F.iloc[i,2])==0:
        if TimetoInt(F.iloc[i,1]) > 2021060100:
            FTR.append(F.iloc[i,1:5])
        if TimetoInt(F.iloc[i,1]) < 2021060100:
            break
print('SMR')
pd.DataFrame(FTR).to_csv("FTR.csv",index=None)
