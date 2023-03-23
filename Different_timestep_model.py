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
# Ts = Megi                    #跑單場
# AD = ImportCSV("AD",None)
# F = CaseDict(AD)
Tr.columns = init_input
Ts.columns = init_input

df0 = pd.DataFrame()
df1 = pd.DataFrame()
for columnName in Tr.columns: 
    for i in range(12,-1,-1):
        if i == 0:
            df0[columnName] = Tr[columnName].shift(i)
        else:
            df0[columnName+'t-'+str(i)] = Tr[columnName].shift(i)

for columnName in Ts.columns: 
    for i in range(12,-1,-1):
        if i == 0:
            df1[columnName] = Ts[columnName].shift(i)
        else:
            df1[columnName+'t-'+str(i)] = Ts[columnName].shift(i)
select_feature = input
# select_feature = ['SMOutflow', 'Tide', 'FTOutflowt-2', 'Tidet-8', 'Tidet-6', 'Tidet-5', 'Tidet-4', 'Tidet-3', 'Tidet-1', 'WaterLevelt-1','WaterLevel']

Tr = df0[12:]
Ts = df1[12:]
Tr = Tr[select_feature]
Ts = Ts[select_feature]
# 預報值
Fors = Ts[1:]
# RandomList = []
# RandomList2 = []
# for i in range(len(Fors["Tide"])):
#     RandomList.append(round(random.uniform(0.8,1.2),1))
#     RandomList2.append(round(random.uniform(0.8,1.2),1))
# Fors["SMOutflow"] = Fors["SMOutflow"]*RandomList
# Fors["Tide"] = Fors["Tide"]*RandomList2

# 檔名
subtitle = "0319select_feature"
# MSF 
TPB_Data = DataSet()
TPB_Data.TrainingData = Tr
TPB_Data.TestData =  Ts

# 載入模型
# df = ImportCSV(f"./LSTM/3/0225tt/(IndexAll)",None)
# df = df.sort_values(by=['CC'], ascending=False)
# for model in df["model"][80:100]:
num = 0
for C in [1,5,10,15,20]:
    for gamma in [0.125, 0.125*2, 0.125*4]:
        num +=1 
        # for num, endTimeList in enumerate([[0]], start=1):
        for TimeStep in [6]:
            NPara = Para()
            NPara.TStep = TimeStep
            # NPara.shape = sum(timeList)+sum(endTimeList)+len(timeList)
            NPara.TPlus = 1
            NPara.FeatureN = len(input) #7
            Npr = pr()
            # X_train, Y_train, X_test, Y_test = Npr.SameProcessing(Dset=TPB_Data, Para=NPara)
            X_train, Y_train, X_test, Y_test = Npr.OneRowProcessing(Dset=TPB_Data, Para=NPara)
            NPara.inputShape = (X_train.shape[1],X_train.shape[2])
            # print (pr._Normalization) 
            ## 輸入項預報值正規化
            Fors = Npr._ForcastNormal(Fors)
            #[64,64,64,64],[128,128,8],[16,16,16,16],[8,8,8],[64,128,256,128,64],[8,16,32],[32,32,32,20],[8,16]
            #[256,128,64],[32,32,32]
            for layer in [[64, 128, 64]]:   #[64,128,64],[128,256,128],[128,256]
                for name in ["SVM"]: #,"RNN","SVM","Seq2Seq" 
                    NPara.Layer = layer
                    NPara.ModelName = name
                    NPara.FeatureN = len(input) #7
                    path = f"{name}\{TimeStep}\{subtitle}"
                    savePath = f"{len(layer)}{layer[0]}({num})"
                    # for para in []
                    if name == "SVM":
                        GP = GPara()
                        GP.gamma = gamma
                        GP.C = C
                        newModel=machineLearning(name, GP)
                        history, fitModel = FittingModel(newModel,name, X_train, Y_train, GP)
                        CheckFile(f"{path}")
                    else:
                        GP = GPara()
                        GP.activate = 'relu'
                        GP.btcsz = 16 #16 或 32
                        GP.opt =  'rmsprop' #'rmsprop' 'adam'
                        GP.epochs = 150
                        GP.loss = "mae" #mae msle
                        GP.lr = 0.00001
                        newModel = deepLearning(name, NPara, GP, layer)
                        history, fitModel = FittingModel(newModel,name,X_train, Y_train, GP)
                        CheckFile(f"{path}")
                        plotHistory(history,f"{path}\{savePath}")
                        ##存檔
                        CheckFile(f"saved_model\{name}")
                        fitModel.save(f'saved_model\{path}\{savePath}.h5')
                    forcasting = Prediction(fitModel,name, X_test)
                    forcastingTr = Prediction(fitModel,name, X_train)
                    ##反正規　畫圖
                    Y_Inv = Npr._InverseCol(Y_test)
                    F_Inv = Npr._InverseCol(forcasting) 
                    PlotResult = ForcastCurve( name ,200, NPara, F_Inv, Y_Inv, GP, subtitle, "test", fileName=f"{path}\{savePath}")
                    
                    Yr_Inv = Npr._InverseCol(Y_train)
                    Fr_Inv = Npr._InverseCol(forcastingTr) 
                    PlotResult = ForcastCurve( name ,200, NPara, Fr_Inv, Yr_Inv, GP ,subtitle, "train", fileName=f"{path}\{savePath}(train)")
                    

                    # #鬍鬚圖
                    # Single = [Y_Inv[20:50]]
                    # time = 0
                    # ForsOne = Fors[20:]
                    # event = X_test[20:50]
                    # for x in range(len(event)):
                    #     temp = []
                    #     time+=1
                    #     # 刪掉第一個值
                    #     Xtest = np.reshape(event[x], (1, 1, event[0].shape[0]))
                    #     # Ytest = Y_test[x]
                    #     # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
                    #     forcasting = Prediction(fitModel, NPara.ModelName, Xtest) 
                    #     F_Inv = Npr._InverseCol(forcasting) 
                    #     temp.append(np.reshape(F_Inv,(1))[0])
                    #     # print(time)
                    #     for i in range(time+1,20): 
                    #     # for i in range(time+1,len(X_test)-2):
                    #         ForsOne[x][-1] = forcasting
                    #         new_x = np.reshape(ForsOne[x], (1, 1, ForsOne[x].shape[0]))
                    #         forcasting = Prediction(fitModel, NPara.ModelName, new_x)
                    #         F_Inv = Npr._InverseCol(forcasting) 
                    #         temp.append(np.reshape(F_Inv,(1))[0])
                    #     N = 0
                    #     while N<x :    
                    #         temp.insert(0,"")
                    #         N+=1 
                    #     Single.append(temp)
                    # df = pd.DataFrame( Single )
                    # DF2CSV(df, f"{path}\{savePath}")
                    # dff = df.T[:20][:19]
                    # plotMegiMSF(dff,f"{path}\{savePath}")

