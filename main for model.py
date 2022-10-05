from Model.trainModel import *
from Control.ImportData import *
from Control.CaseSplit import *
import numpy as np


Tr = ImportCSV("Train",None)
Ts =  ImportCSV("Test",None)
# AD = ImportCSV("AD",None)
# F = CaseDict(AD)

# Tr = TrainingData[~TrainingData['#1'].str.contains('#')] #找出不含'#'的row
# TrainingData[~TrainingData['#1'].isin(['#'])]  #找出不等於'#'的row
Tr = Tr[~Tr['#1'].str.contains('#')]
Ts = Ts[~Ts['#16'].str.contains('#')]
# F = {'first':Tr}

# 網格搜尋法參數調整
activate = ['relu','tanh']
opt = ['rmsprop', 'adam']
epochs = [50]
# hl1_nodes = np.array([1, 10, 50])
btcsz = [1,16,32]
loss = ['mse','msle']

# 場次選擇
# Tr = np.arange(2,41)
# Ts = np.arange(42,49)
# Global para
for i in [1,3,6]:
    NPara = Para()
    NPara.ModelName = 'SVM' # lstm
    NPara.TStep = 3  
    NPara.TPlus = i  ##預測 T+1 T+3 T+6 
    NPara.FeatureN = 7
    NPara.ParamGrid = dict(optimizer=opt, nb_epoch=epochs,batch_size=btcsz)
    NPara.Scoring = "neg_mean_absolute_error"

    GP = GPara()
    GP.activate = 'relu'
    GP.btcsz = 32 #16 或 32
    GP.opt =  'rmsprop' #'rmsprop'
    GP.epochs = 100
    GP.loss = "msle"

    TPB_Data = DataSet()
    TPB_Data.TrainingData = Tr
    TPB_Data.TestData =  Ts
    NewL = L()
    NewL.Define(NPara,TPB_Data,GP)
    NewL.DataProcessing()
    NewL.ModelSetting(False)
    NewL.PlotResult()


# 1.把訓練測試集從字典合成
# 2.跑模式
# # 3.預測單場
# a = np.array(F[1])
# for i in [Tr]:
#     np.concatenate(np.array([F[i]], ignore_index=True))
# b = np.array(F[41])
# for i in [Ts]:
#     b.append(np.array([F[i]], ignore_index=True))

