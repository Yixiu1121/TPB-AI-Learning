from Model.TPBdata import *
import pandas as pd


def ImportCSV(fileName :str, colName : list):
    return pd.read_csv(fileName+".csv",usecols=colName)

def DF2CSV(df,fileName):
    # "預設沒有index & header"
    return df.to_csv(fileName+".csv",index=False, header = False)

def DF2List(df):
    return df.values.tolist()

def DataList(data,datalist:list,paraNum:int):
    data = [datalist[i][paraNum] for i in range(len(data))]
    return data

    
if __name__ == "__main__":
    print("error")
    SM = Reservoir()
    Property = [SM.Date, SM.Inflow, SM.Outflow]
    ShimenList = DF2List(ImportCSV("石門.csv",["水情時間","進流量(cms)","出流量(cms)"]))
    n = len(Property)
    for i in Property:
        DataList(i,ShimenList,-n)
        n -=1

    FT = Reservoir()
    Property = [FT.Date, FT.Outflow]
    FeitsuiList = DF2List(ImportCSV("翡翠.csv",["水情時間","出流量(cms)"]))
    n = len(Property)
    for i in Property:
        DataList(i,ShimenList,-n)
        n -=1
