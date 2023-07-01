from doctest import testfile
import imghdr
# from Model.TPBdata import *
import pandas as pd
import os

def ImportCSV(fileName :str, colName : list):
    return pd.read_csv(fileName+".csv",usecols=colName)

def DF2CSV(df,fileName):
    # "預設沒有index & header"
    return df.to_csv(fileName+".csv",index = True, header = False)

def DF2CSVH(df,fileName):
    # "預設沒有index & 有header"
    return df.to_csv(fileName+".csv",index = False, header = True)

def DF2List(df):
    return df.values.tolist()

def DataList(data,datalist:list,paraNum:int):
    data = [datalist[i][paraNum] for i in range(len(data))]
    return data

def CheckFile(path):
    "確認資料夾"
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
    

if __name__ == "__main__":
    import csv
    path = "C:\\Users\\309\\Documents\\台北橋石門入流量預報_雨量預報\\57.ShihmenWRPIQ_L_LSTM_SPM_梅姬.csv"
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        lis_reader = []
        for row in reader:
            if not row or row=="":
                continue
            else: lis_reader.append(row)
    print( lis_reader[0])
    # print("error")
    # SM = Reservoir()
    # Property = [SM.Date, SM.Inflow, SM.Outflow]
    # ShimenList = DF2List(ImportCSV("石門.csv",["水情時間","進流量(cms)","出流量(cms)"]))
    # n = len(Property)
    # for i in Property:
    #     DataList(i,ShimenList,-n)
    #     n -=1

    # FT = Reservoir()
    # Property = [FT.Date, FT.Outflow]
    # FeitsuiList = DF2List(ImportCSV("翡翠.csv",["水情時間","出流量(cms)"]))
    # n = len(Property)
    # for i in Property:
    #     DataList(i,ShimenList,-n)
    #     n -=1

