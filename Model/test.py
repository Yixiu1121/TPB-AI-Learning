from tensorflow.keras import Sequential, losses, optimizers
from tensorflow.keras.layers import LSTM,Dropout,Dense,Flatten,RNN,SimpleRNN,RepeatVector,TimeDistributed,Input,Conv1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn import svm
import numpy as np
import sys
sys.path.append("C:/Users/309/Documents/GitHub/TPB Code/TPB-AI-Learning")

from Model.Processing import dataFunction
from Control.ImportData import *
import matplotlib.pyplot as plt
from Model.trainModel import *
from Model.Processing import Processed_Data as pr
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

from pandas import Series
from pandas import DataFrame
# load dataset
# series = Series.from_csv('seasonally_adjusted.csv', header=None)
# reframe as supervised learning
for i in range(12,0,-1):
    dataframe['t-'+str(i)] = Tr.shift(i)
    dataframe['t'] = Tr.values
    print(dataframe.head(13))
    dataframe = dataframe[13:]
# save to new file
dataframe.to_csv('lags_12months_features.csv', index=False)

