##select the starttime and endtime
##new a event

# from unittest import result
import datetime
from Model.TPBdata import *

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

# print(datetime.datetime.strptime("2021/6/7 01:00",'%Y/%m/%d %H:%M'))