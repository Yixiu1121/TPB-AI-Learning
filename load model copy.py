from Model.trainModel import *
from Control.ImportData import *
from Control.CaseSplit import *
from Model.Processing import Processed_Data as pr
from Model.ModelSetting import *
from Model.MSF import *
import numpy as np
import tensorflow as tf
import random

# 因子調整
init_input = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']  #8個
input = ['TPB','SMOutflow','FTOutflow','Tide','WaterLevel']   #7個
Tr = ImportCSV("Train",None) #SM FT TPB SMInflow SMOutflow FTOutflow Tide WL
Ts =  ImportCSV("Test",None)
Megi = ImportCSV("Megi",None)
Dujan = ImportCSV("Dujan",None)
Tr = Tr[~Tr['#1'].str.contains('#')]
Ts = Ts[~Ts['#16'].str.contains('#')]
Ts = Megi                    #跑單場
# AD = ImportCSV("AD",None)
# F = CaseDict(AD)
Tr.columns = init_input
Ts.columns = init_input

# 差值
# Tr['WaterLevel'] = Tr['WaterLevel']*100-Tr['Tide']  #差值
# Ts['WaterLevel'] = Ts['WaterLevel']*100-Ts['Tide']  #差值
Tr = Tr[input]
Ts = Ts[input]

# 預報值
Fors = Ts[input]
# RandomList = []
# RandomList2 = []
# for i in range(len(Fors["Tide"])):
#     RandomList.append(round(random.uniform(0.8,1.2),1))
#     RandomList2.append(round(random.uniform(0.8,1.2),1))
# Fors["SMOutflow"] = Fors["SMOutflow"]*RandomList
# Fors["Tide"] = Fors["Tide"]*RandomList2

# 檔名
subtitle = "0225tt_前20"
# MSF 
TPB_Data = DataSet()
TPB_Data.TrainingData = Tr
TPB_Data.TestData =  Ts
timeList = [1, 3, 3, 12, 6]
# 載入模型
# df = ImportCSV(f"./LSTM/3/0225tt/(IndexAll)",None)
# df = df.sort_values(by=['CC'], ascending=False)
# for model in df["model"][80:100]:
for model in ["432(1)"]:
    importPath = f"LSTM/12/0308t1/{model}"
    load_model = tf.keras.models.load_model(f'saved_model/{importPath}.h5')
    NPara = Para()
    NPara.TStep = 3
    NPara.TPlus = 1
    NPara.ModelName = "LSTM"
    NPara.FeatureN = len(input) #7
    NPara.TStepList = timeList
    Npr = pr()
    X_train, Y_train, X_test, Y_test = Npr.DataProcessing(Dset=TPB_Data, Para=NPara)
    Fors = Npr._ForcastNormal(Fors)

    forcasting = Prediction(load_model, NPara.ModelName, X_train) 
    Y_Inv = Npr._InverseCol(Y_train)
    F_Inv = Npr._InverseCol(forcasting) 
    PlotResult = ForcastCurve(200, NPara, F_Inv, Y_Inv, "",subtitle, fileName="load")
    
    #鬍鬚圖
    Single = [Y_Inv[20:50]]
    time = 0
    ForsOne = Fors[20:]
    ForsOne = ForsOne[max(timeList)+1:]
    event = X_train[20:50]
    for x in range(len(event)):
        temp = []
        time+=1
        Xtest = np.reshape(event[x], (1, 1, event[x].shape[1]))
        # Ytest = Y_test[x]
        # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
        forcasting = Prediction(load_model, NPara.ModelName, Xtest) 
        F_Inv = Npr._InverseCol(forcasting) 
        temp.append(np.reshape(F_Inv,(1))[0])
        # print(time)
        for i in range(time+1,20): 
        # for i in range(time+1,len(X_test)-2):
            new_x, new_y = msf1D(timeList = NPara.TStepList, Fors = ForsOne[x], X_test = Xtest , Y_test = "" , forcasting=forcasting)
            Xtest = new_x
            # Ytest = new_y
            # print(new_x[0], new_y[0])
            # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
            forcasting = Prediction(load_model, NPara.ModelName, new_x)
            # print("X=",new_x[0],"Y=",new_y[0],forcasting[0])
            F_Inv = Npr._InverseCol(forcasting) 
            temp.append(np.reshape(F_Inv,(1))[0])
        N = 0
        while N<x :    
            temp.insert(0,"")
            N+=1 
        Single.append(temp)
    df = pd.DataFrame( Single )
    DF2CSV(df, f"{importPath}(train)")
    dff = df.T[:20][:19]
    plotMegiMSF(dff,f"{importPath}(train)")

