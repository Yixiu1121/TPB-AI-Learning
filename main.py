from Control.CalcRain import *
from Control.CaseSplit import *
from Control.ImportData import *
from Model.TPBdata import *
from Source.WLdata import *

if __name__ == "__main__":
    #1. 找出List資料(水庫、雨量站、潮位)
    #2. 創建新的Dictionoary(時間:台北橋)
    #3. 用循環去Run Calc的Function
    #4. 轉換成Excel

    # SMLst = DF2List(ImportCSV("SMR_all",['INFO_DATE','Q_IN','Q_OUT']))
    # FTLst = DF2List(ImportCSV("FTR_all",['INFO_DATE','Q_IN','Q_OUT']))
    # TideLst = DF2List(ImportCSV("Danshui(all)",['D','mm']))
    # WaterLevelLst =  DF2List(ImportCSV("P04ALL",['INFO_DATE','H']))
    # CaseLst = DF2List(ImportCSV("Event_aLL",['caseNum', 'startTime', 'endTime']))

    # rainfallLst=[]
    result={}
    
    #家浩翡翠資料
    FTLst = DF2List(ImportCSV("FTR_plus",['DDateTime','outflow','st1','st2','st3','st4','st5','st6']))
    CaseLst = DF2List(ImportCSV("Event_old",['caseNum', 'startTime', 'endTime']).dropna(axis=0,how='all'))
    

    ## 注意時間欄的形式
    # for tide in TideLst:
    #     '2022/1/1 00:00'
    #     T = Tide()
    #     T.Date = tide[0]
    #     T.Tide = tide[1]
    #     CalcTide(T,result)

    # for rainfall in rainfallLst:
    #     T = Rainfall()
    #     T.Date = rainfall[0]
    #     T.Rainfall = rainfall[1]
    #     CalcRainfall(rainfall,result)

    # for wl in WaterLevelLst:
    #     T = WaterLevel()
    #     T.Date = wl[0]
    #     T.WaterLevel = wl[1]
    #     CalcWaterLevel(T,result)

    ##24小時制
    # for reservoir in SMLst:
    #     '2022/1/1 00:00'
    #     T = Reservoir()
    #     T.Date = reservoir[0]
    #     T.Inflow = reservoir[1]
    #     T.Outflow = reservoir[2]
    #     T.Name = "石門"
    #     CalcReservoir(T,result)
    
    # for reservoir in FTLst:
    #     "2022/9/15 3:00 PM"
    #     T = Reservoir()
    #     T.Date = Time24(reservoir[0])
    #     T.Inflow = reservoir[1]
    #     T.Outflow = reservoir[2]
    #     T.Name = "翡翠"
    #     CalcReservoir(T,result)


# print(result["2021/6/7 19:00"].Tide)
# print(pd.DataFrame.from_dict(result,orient='index'))

## 事件
# CaseNum = len(CaseLst)
# Event = [['Date','Tide','ShihmenInflow','ShihmenOutflow','FeitsuiOutflow','WaterLevel']]
# for case in CaseLst:
#     T = CreateCaseLST(case[0],case[1],case[2])
#     Event.append(["#"+str(T.Casenum),T.StrTime,T.EndTime])
#     for date in result.keys():
#         if DateCompare(date,T.StrTime,T.EndTime):
#             R = result[date]
#             p = []
#             for property in [R.Date, R.Tide,R.ShihmenInflow,R.ShihmenOutflow,R.FeitsuiOutflow,R.WaterLevel]:    
#                 p.append(property)
#             Event.append(p)
# DF2CSV(pd.DataFrame(Event),'Event')


CaseNum = len(CaseLst)
Event = [['Date','FeitsuiOutflow','翡翠','碧湖','九芎根','十三股',	'坪林',	'太平']]
for case in CaseLst:
    T = CreateCaseLST(case[0],case[1],case[2])
    Event.append(["#"+str(T.Casenum),T.StrTime,T.EndTime])
    print(T.Casenum)
    for i in FTLst:
        # print(i)
        if DateCompare(i[0],T.StrTime,T.EndTime)==True:
            Event.append(i)
        if DateCompare(i[0],T.StrTime,T.EndTime)=="bigger":
            break
DF2CSV(pd.DataFrame(Event),'FTevent')




