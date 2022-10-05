##select the starttime and endtime
##new a event

# from unittest import result
import datetime
from Model.TPBdata import *
import pandas as pd

def CreateCaseLST(num,STR,END):
    T= Case()
    T.Casenum = num 
    T.StrTime = STR
    T.EndTime = END
    return T

def DateCompare(compareDate,strTime,endTime):
    "2021/6/7 19:00"
    cD = datetime.datetime.strptime(compareDate,'%Y/%m/%d %H:%M')
    sT = datetime.datetime.strptime(strTime,'%Y/%m/%d %H:%M')
    eT = datetime.datetime.strptime(endTime,'%Y/%m/%d %H:%M')
    if cD >= sT and cD <= eT:
        return True

def CaseDict(AD):
    "場次字典"
    F = {}
    temp = []
    last = 1
    for i in range(len(AD['#1'])):
        if AD['#1'][i][0]!='#':
            temp.append((AD.iloc[i,:]))
        elif AD['#1'][i][0]=='#':
            # print(pd.DataFrame(temp))
            F[last] = pd.DataFrame(temp)
            temp = []
            last +=1
    F[last] = pd.DataFrame(temp)
    return F
# def CaseDict(AD):
#     "場次字典"
#     F = {}
#     temp = []
#     for i in range(len(AD['#1'])):
#         if AD['#1'][i][0]!='#':
#             temp.append((AD.iloc[i,:]))
#         elif AD['#1'][i][0]=='#':
#             # print(pd.DataFrame(temp))
#             F[int(AD['#1'][i][1:])-1] = pd.DataFrame(temp)
#             temp = []
#             last = AD['#1'][i][1:]
#     F[int(last)] = pd.DataFrame(temp)
#     return F
# print(datetime.datetime.strptime("2021/6/7 01:00",'%Y/%m/%d %H:%M'))