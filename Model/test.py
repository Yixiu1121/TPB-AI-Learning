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
batch_size = 32
in_steps = 1
in_vector = 26 #因子數
out_vector = 12
NPara = Para()
timeList = [1,3,3,12,6]
NPara.TStep = timeList
NPara.shape = sum(timeList)+1
NPara.TStepList = timeList
NPara.TPlus = 1
NPara.FeatureN = len(input) #7
model = Sequential()
model.add(LSTM(128,batch_input_shape=(batch_size, in_steps ,in_vector), stateful=True))
model.add(RepeatVector(20))
model.add(TimeDistributed(Dense(out_vector, activation='relu')))
print(model.summary())
model.compile(loss='mse', optimizer='adam')
TPB_Data = DataSet()
TPB_Data.TrainingData = Tr
TPB_Data.TestData =  Ts
Npr = pr()
X_train, Y_train, X_test, Y_test = Npr.DataProcessing(Dset=TPB_Data, Para=NPara)
newy = []
for i in range(len(Y_train)-11):
    newy.append(Y_train[i:i+12])
new_Y_test = []
for i in range(len(Y_test)-11):
    newy.append(Y_test[i:i+12])
history = model.fit(X_train, newy, epochs = 200, batch_size = batch_size ,validation_split=0.2)
result = model.predict(X_test, batch_size = 32, verbose=0)

