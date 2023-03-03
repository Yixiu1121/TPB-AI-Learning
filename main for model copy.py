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
input = ['Shihimen','Feitsui','TPB','SMOutflow','FTOutflow','Tide','WaterLevel']   #7個
Tr = ImportCSV("Train",None) #SM FT TPB SMInflow SMOutflow FTOutflow Tide WL
Ts =  ImportCSV("Test",None)
# Megi = ImportCSV("Megi",None)
# Dujan = ImportCSV("Dujan",None)
Tr = Tr[~Tr['#1'].str.contains('#')]
Ts = Ts[~Ts['#16'].str.contains('#')]
# Ts = Dujan                    #跑單場
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
subtitle = "0303"
# 網格搜尋法參數調整
activate = ['relu','tanh']
opt = ['rmsprop', 'adam']
epochs = [2]
# hl1_nodes = np.array([1, 10, 50])
btcsz = [1,16,32]
loss = ['mse','msle']
timeList = [3,3,3,3,3,6,3]
# MSF 
TPB_Data = DataSet()
TPB_Data.TrainingData = Tr
TPB_Data.TestData =  Ts


# 訓練模型 
num = 0
for TimeStep in [5]:
    NPara = Para()
    NPara.TStep = TimeStep
    NPara.TPlus = 1
    NPara.FeatureN = len(input) #7
    Npr = pr()
    X_train, Y_train, X_test, Y_test = Npr.DataProcessing(Dset=TPB_Data, Para=NPara)
    # print (pr._Normalization) 
    ## 輸入項預報值正規化
    Fors = Npr._ForcastNormal(Fors)
    #[64,64,64,64],[128,128,8],[16,16,16,16],[8,8,8],[64,128,256,128,64],[8,16,32],[32,32,32,20],[8,16]
    #[256,128,64],[32,32,32]   
    for name in ["SVM"]: #,"RNN","SVM","Seq2Seq"
        NPara.ModelName = name
        NPara.FeatureN = len(input) #7
        path = f"{name}\{TimeStep}\{subtitle}"
        savePath = f"({num})"
        # for para in []:
        if name == "SVM":
            newModel=machineLearning(name)
            GP = ""
            history, fitModel = FittingModel(newModel,name, X_train, Y_train, GP)
        forcasting = Prediction(fitModel,name, X_test)
        
        ##反正規　畫圖
        Y_Inv = Npr._InverseCol(Y_test)
        F_Inv = Npr._InverseCol(forcasting) 
        PlotResult = ForcastCurve(NPara, F_Inv[:200], Y_Inv[:200], GP ,subtitle, fileName=savePath)

        #鬍鬚圖
        Single = [Y_Inv[30:60]]
        time = 0
        ForsOne = Fors[30:60]
        ForsOne = ForsOne[7:]
        event = X_test[30:60]
        for x in range(len(event)):
            temp = []
            time+=1
            Xtest = np.reshape(event[x], (1, 1, event[x].shape[1]))
            # Ytest = Y_test[x]
            # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
            forcasting = Prediction(fitModel, NPara.ModelName, Xtest) 
            F_Inv = Npr._InverseCol(forcasting) 
            temp.append(np.reshape(F_Inv,(1))[0])
            # print(time)
            for i in range(time+1,20): 
            # for i in range(time+1,len(X_test)-2):
                new_x, new_y = msf1D(Fors = ForsOne[x], X_test = Xtest , Y_test = "" , forcasting=forcasting, time=i, TStep = NPara.TStep)
                Xtest = new_x
                # Ytest = new_y
                # print(new_x[0], new_y[0])
                # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
                forcasting = Prediction(fitModel, NPara.ModelName, new_x)
                # print("X=",new_x[0],"Y=",new_y[0],forcasting[0])
                F_Inv = Npr._InverseCol(forcasting) 
                temp.append(np.reshape(F_Inv,(1))[0])
            N = 0
            while N<x :    
                temp.insert(0,"")
                N+=1 
            Single.append(temp)
        df = pd.DataFrame( Single )
        DF2CSV(df, f"{path}\{savePath}")
        dff = df.T[:20][:19]
        plotMegiMSF(dff,f"{path}\{savePath}")

                
                
# """
##load多步階
    # for i in range(2,13):
    #     new_x, new_y = msf(Fors, X_test, Y_test, forcasting, time=i, TStep = NPara.TStep)
    #     X_test = new_x
    #     Y_test = new_y
    #     # print(new_x[0], new_y[0])
    #     forcasting = Prediction(new_model, NPara.ModelName, new_x)
    #     # print("X=",new_x[0],"Y=",new_y[0],forcasting[0])
    #     Y_Inv = Npr._InverseCol(new_y)
    #     F_Inv = Npr._InverseCol(forcasting) 
    #     PlotResult = ForcastCurve(NPara, F_Inv, Y_Inv, "",subtitle, fileName=f"MSFload{str(i)}")

##new多步階
                
                # d={"RMSE":[]}
                # for i in range(2,13):
                #     new_x, new_y = msf(Fors, X_test, Y_test, forcasting, time=i, TStep = NPara.TStep)
                #     X_test = new_x
                #     Y_test = new_y
                #     # print(new_x[0], new_y[0])
                #     forcasting = Prediction(fitModel, name, new_x)
                #     # print("X=",new_x[0],"Y=",new_y[0],forcasting[0])
                #     Y_Inv = Npr._InverseCol(new_y)
                #     F_Inv = Npr._InverseCol(forcasting) 
                #     PlotResult = ForcastCurve(NPara, F_Inv, Y_Inv, GP ,subtitle, fileName=f"MSF{str(i)+str(layer[0])+str(len(layer))+subtitle}")
                #     MSFForcastCurve(d, i, Y_Inv, F_Inv )
                # dictCSV(d, NPara, subtitle, fileName=f"MSF{str(i)+str(layer[0])+str(len(layer))+subtitle}" )
