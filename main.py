from Control.CalcRain import *


if __name__ == "__main__":
    #1. 找出List資料(水庫、雨量站、潮位)
    #2. 創建新的Dictionoary(時間:台北橋)
    #3. 用循環去Run Calc的Function
    #4. 轉換成Excel
    #題目：1. 找List的Function
    #      2. 輸入至Excel

    
    rainfallLst=[]
    result={}
    for rainfall in rainfallLst:
        CalcRainfall(rainfall,result)
    