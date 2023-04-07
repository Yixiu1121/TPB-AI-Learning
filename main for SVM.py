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
subtitle = "extendWL for SMO"
name = "SVM"
# timeList = [6,6,6,12,12,12,12]
# shape = sum(timeList)+len(timeList)
# MSF 
TPB_Data = DataSet()
TPB_Data.TrainingData = Tr
TPB_Data.TestData =  Ts
# input = ['TPB','SMInflow','SMOutflow','FTOutflow','WaterLevel']   #7個/
endTimeList=[0,0,0,0,0] #取到t-1, t
# Fors["SMOutflow"] = Fors["SMOutflow"].shift(2) #下移一格 讓SMOutFlow 多步階從t-1開始
# Fors["SMOutflow"] = Fors["SMOutflow"].shift(-1) #上移一格 讓SMOutFlow 多步階從t+1開始
# Fors["SMInflow"] = Fors["SMInflow"].shift(-1) #上移一格 讓SMInflow 多步階從t+1開始
# 訓練模型 
num = 4
epsilon = 2**-4 #0.00390625(2^-8) 0.0009765625(2^-10)
degree = 0
C = 4
gamma = 0.5
# SMI = 3
# layer = [64,128,64]
for T in [3]:
    for SMO in [3]:
        for FT in [3]:
                # for num, endTimeList in enumerate([[0]], start=1):
                for WL in [8]:
                    num +=1 
                    TimeStep = 3
                    NPara = Para()
                    NPara.TStep = TimeStep
                    # NPara.shape = sum(timeList)+sum(endTimeList)+len(timeList)
                    R = 1
                    NPara.TStepList = [R, SMO, FT, T, WL]
                    NPara.EDTStepList = endTimeList
                    NPara.TPlus = 1
                    NPara.FeatureN = len(input) #7
                    Npr = pr()
                    # X_train, Y_train, X_test, Y_test = Npr.SameProcessing(Dset=TPB_Data, Para=NPara)
                    X_train, Y_train, X_test, Y_test = Npr.DataProcessing(Dset=TPB_Data, Para=NPara)
                    NPara.inputShape = (X_train.shape[1],X_train.shape[2])
                    # print (pr._Normalization) 
                    ## 輸入項預報值正規化
                    Fors = Npr._ForcastNormal(Fors)
                    NPara.ModelName = name
                    NPara.FeatureN = len(input) #7
                    path = f"{name}\{TimeStep}\{subtitle}"
                    # for para in []
                    
                    GP = GPara()
                    GP.gamma = gamma
                    GP.C = C
                    GP.kernal = 'rbf'
                    GP.epsilon = epsilon  ##越小越好
                    GP.degree = degree
                    savePath = f"({num})"
                    newModel=machineLearning(name, GP)
                    history, fitModel = FittingModel(newModel,name, X_train, Y_train, GP)
                    CheckFile(f"{path}")                              
                    forcasting = Prediction(fitModel,name, X_test)
                    forcastingTr = Prediction(fitModel,name, X_train)
                    ##反正規　畫圖
                    Y_Inv = Npr._InverseCol(Y_test)
                    F_Inv = Npr._InverseCol(forcasting) 
                    PlotResult = ForcastCurve( name, 200, NPara, F_Inv, Y_Inv, GP ,subtitle, "test", fileName=f"{path}\{savePath}")
                    
                    Yr_Inv = Npr._InverseCol(Y_train)
                    Fr_Inv = Npr._InverseCol(forcastingTr) 
                    PlotResult = ForcastCurve( name, 1000, NPara, Fr_Inv, Yr_Inv, GP ,subtitle, "train", fileName=f"{path}\{savePath}(train)")

                    # 鬍鬚圖&每條歷線指標(check t-1, t+1正確) 
                    Ty = ["Dujan", "Megi", "Mitag"] #三場颱風
                    Starttime = [21, 143, 606] #起始點 Dujan21, Megi143  ,Mitag606
                    for ty, starttime in zip(Ty, Starttime):
                        endLength = 25 #往後長度 +12

                        obs = Y_Inv[starttime:starttime+endLength+12]
                        Single = [obs]
                        Time = 0
                        ForsOne = Fors[starttime:]
                        ForsOne = ForsOne[max([R, SMO, FT, T, WL]):] ## t時刻
                        # ForsOne = ForsOne[max(timeList):] ## t時刻
                        event = X_test[starttime:starttime+endLength] #25:50 145:175
                        index12 = {"RMSE":[], "MAE":[], "CC":[], "CE":[]} #算每條歷線index
                        for startPoint in range(len(event)):
                            temp = []
                            Time+=1
                            Xtest = np.reshape(event[startPoint], (1, 1, event[startPoint].shape[1]))
                            forcasting = Prediction(fitModel, NPara.ModelName, Xtest) 
                            F_Inv = Npr._InverseCol(forcasting) 
                            temp.append(np.reshape(F_Inv,(1))[0])
                            
                            # df = pd.DataFrame()
                            recursiveTime = startPoint
                            for i in range(Time+1,Time+1+11): 
                            # for i in range(time+1,len(X_test)-2):
                                recursiveTime +=1
                                new_x, new_y = msf1D(endTimeList = endTimeList, timeList = NPara.TStepList, Fors = ForsOne[recursiveTime], X_test = Xtest , Y_test = "" , forcasting=forcasting)
                                    
                                ## debug
                                # df[f"X_test{i}"] = Xtest.flatten()
                                # df[f"new_x{i}"] = new_x.flatten()
                                # df[f"Fors{i}"] = pd.Series(forsTuple( ForsOne[recursiveTime],forcasting))
                                
                                Xtest = new_x
                                forcasting = Prediction(fitModel, NPara.ModelName, new_x)
                                F_Inv = Npr._InverseCol(forcasting) 
                                temp.append(np.reshape(F_Inv,(1))[0])
                            d = dataFunction.Index(np.array(obs[Time:Time+12]),np.array(temp))
                            index12["RMSE"].append(d["RMSE"])
                            index12["MAE"].append(d["MAE"])
                            index12["CC"].append(d["CC"])
                            index12["CE"].append(d["CE"])
                            # DF2CSV(df.T, "msft+1")
                            N = 0
                            while N < startPoint :    
                                temp.insert(0,"")
                                N+=1 
                            Single.append(temp)
                        df = pd.DataFrame( Single )
                        DF2CSV(df, f"{path}\{savePath}{ty}")

                        plotEventMSF(df.T,f"{path}\{savePath}t",xlength = endLength+12, eventName = ty)
                        DF2CSVH(pd.DataFrame(index12), f"{path}\{savePath}{ty}index")

                            # # t+12
                            # Single = []
                            # Time = 0
                            # ForsOne = Fors
                            # ForsOne = ForsOne[max([R, SMO, FT, T, WL]):] ## t時刻
                            # # ForsOne = ForsOne[max(timeList):] ## t時刻
                            # event = X_test #25:50 145:175
                            # for startPoint in range(len(event)-12):
                            #     temp = []
                            #     Time+=1
                            #     Xtest = np.reshape(event[startPoint], (1, 1, event[startPoint].shape[1]))
                            #     forcasting = Prediction(fitModel, NPara.ModelName, Xtest) 
                            #     F_Inv = Npr._InverseCol(forcasting) 
                            #     temp.append(np.reshape(F_Inv,(1))[0])
                                
                            #     # df = pd.DataFrame()
                            #     recursiveTime = startPoint
                            #     for i in range(11): 
                            #     # for i in range(time+1,len(X_test)-2):
                            #         recursiveTime +=1
                            #         new_x, new_y = msf1D(endTimeList = endTimeList, timeList = NPara.TStepList, Fors = ForsOne[recursiveTime], X_test = Xtest , Y_test = "" , forcasting=forcasting)
                                     
                            #         ## debug
                            #         # df[f"X_test{i}"] = Xtest.flatten()
                            #         # df[f"new_x{i}"] = new_x.flatten()
                            #         # df[f"Fors{i}"] = pd.Series(forsTuple( ForsOne[recursiveTime],forcasting))
                                    
                            #         Xtest = new_x
                            #         forcasting = Prediction(fitModel, NPara.ModelName, new_x)
                            #         F_Inv = Npr._InverseCol(forcasting) 
                            #         temp.append(np.reshape(F_Inv,(1))[0])
                            #     # DF2CSV(df.T, "msft+1")
                            #     Single.append(temp)
                            # df1 = pd.DataFrame( Single )
                            # df2 = pd.DataFrame()
                            # df2["Observation"] = Y_Inv
                            # result = pd.concat([df2,df1], axis=1)
                            # index12 = {"RMSE":[], "MAE":[], "CC":[], "CE":[]}
                            # for i in range(12):
                            #     d = dataFunction.Index(Y_Inv[i:len(df1[i])+i],df1[i])
                            #     index12["RMSE"].append(d["RMSE"])
                            #     index12["MAE"].append(d["MAE"])
                            #     index12["CC"].append(d["CC"])
                            #     index12["CE"].append(d["CE"])
                            # DF2CSVH(result, f"{path}\{savePath}t12")
                            # DF2CSVH(pd.DataFrame(index12), f"{path}\{savePath}t12index")
                            # dff = df.T[:30][:19]
                            # plotMegiMSF(dff,f"{path}\{savePath}t+12")


                            
                            # # 鬍鬚圖(error: t-1的放錯)
                            # Single = [Y_Inv[25:50]]
                            # Time = 0
                            # ForsOne = Fors[25:]
                            # ForsOne = ForsOne[max([R, SMO, FT, T, WL]):] ## t時刻
                            # # ForsOne = ForsOne[max(timeList):] ## t時刻
                            # event = X_test[25:50] #25:50 145:175
                            # for startPoint in range(len(event)):
                            #     temp = []
                            #     Time+=1
                            #     Xtest = np.reshape(event[startPoint], (1, 1, event[startPoint].shape[1]))
                            #     forcasting = Prediction(fitModel, NPara.ModelName, Xtest) 
                            #     F_Inv = Npr._InverseCol(forcasting) 
                            #     temp.append(np.reshape(F_Inv,(1))[0])
                                
                            #     # df = pd.DataFrame()
                            #     recursiveTime = startPoint
                            #     for i in range(Time+1,20): 
                            #     # for i in range(time+1,len(X_test)-2):
                            #         recursiveTime +=1
                            #         new_x, new_y = msf1D(endTimeList = endTimeList, timeList = NPara.TStepList, Fors = ForsOne[recursiveTime], X_test = Xtest , Y_test = "" , forcasting=forcasting)
                                     
                            #         ## debug
                            #         # df[f"X_test{i}"] = Xtest.flatten()
                            #         # df[f"new_x{i}"] = new_x.flatten()
                            #         # df[f"Fors{i}"] = pd.Series(forsTuple( ForsOne[recursiveTime],forcasting))
                                    
                            #         Xtest = new_x
                            #         forcasting = Prediction(fitModel, NPara.ModelName, new_x)
                            #         F_Inv = Npr._InverseCol(forcasting) 
                            #         temp.append(np.reshape(F_Inv,(1))[0])
                            #     # DF2CSV(df.T, "msft+1")
                            #     N = 0
                            #     while N < startPoint :    
                            #         temp.insert(0,"")
                            #         N+=1 
                            #     Single.append(temp)
                            # df = pd.DataFrame( Single )
                            # DF2CSV(df, f"{path}\{savePath}")
                            # dff = df.T[:30][:19]
                            # plotMegiMSF(dff,f"{path}\{savePath}")
                            


                    # #同步長鬍鬚圖
                    # Single = [Y_Inv[20:50]]
                    # time = 0
                    # ForsOne = Fors[20:]
                    # ForsOne = ForsOne[max(timeList)+1:] ## t+1時刻
                    # # ForsOne = ForsOne[max(timeList):] ## t時刻
                    # event = X_test[20:50]
                    # for x in range(len(event)):
                    #     temp = []
                    #     time+=1
                    #     Xtest = event[x]
                    #     # Ytest = Y_test[x]
                    #     # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
                    #     forcasting = Prediction(fitModel, NPara.ModelName, np.reshape(Xtest, (1, Xtest.shape[0], Xtest.shape[1]))) 
                    #     F_Inv = Npr._InverseCol(forcasting) 
                    #     temp.append(np.reshape(F_Inv,(1))[0])
                    #     # print(time)
                    #     for i in range(time+1,20): 
                    #     # for i in range(time+1,len(X_test)-2):
                    #         new_x, new_y = msf( Fors = ForsOne[x], X_test = Xtest , Y_test = "" , forcasting=forcasting)
                    #         Xtest = new_x
                    #         # Ytest = new_y
                    #         # print(new_x[0], new_y[0])
                    #         # forcasting = Prediction(new_model, NPara.ModelName, np.reshape(new_x, (1, 4, 7)))
                    #         forcasting = Prediction(fitModel, NPara.ModelName, np.reshape(new_x, (1, new_x.shape[0], new_x.shape[1])))
                    #         # print("X=",new_x[0],"Y=",new_y[0],forcasting[0])
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

