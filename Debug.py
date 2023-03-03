# from Control.ImportData import *
# from Control.CaseSplit import *
# from Model.trainModel import *
# from Model.Processing import Processed_Data as pr
# init_input = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']  #8個
# input = ['Shihimen','Feitsui','TPB','SMOutflow','FTOutflow','Tide','WaterLevel']   #7個
# Tr = ImportCSV("Train",None) #SM FT TPB SMInflow SMOutflow FTOutflow Tide WL
# Ts =  ImportCSV("Test",None)
# Megi = ImportCSV("Megi",None)
# Tr = Tr[~Tr['#1'].str.contains('#')]

# Ts = Megi
# Tr.columns = init_input
# Ts.columns = init_input

# Tr = Tr[input]
# Ts = Ts[input]

# Fors = Ts[input]

# TPB_Data = DataSet()
# TPB_Data.TrainingData = Tr
# TPB_Data.TestData =  Ts

# NPara = Para()
# NPara.TStep = 3
# NPara.TPlus = 1
# Npr = pr()
# X_train, Y_train, X_test, Y_test = Npr.DataProcessing(Dset=TPB_Data, Para=NPara)
# Fors = Npr._ForcastNormal(Fors)
from Control.ImportData import *
# header = ["model","RMSE", "MAE", "CE", "CC"]
# Index = []
# for layer in ["38","416","464","3128"]:
#       for num in range(1,50):
#             model = []
#             Tr = ImportCSV(f"./LSTM/3/0225tt/{layer}({num})index",None)
#             model.append(f"{layer}({num})")
#             for index in ["RMSE", "MAE", "CE", "CC"]:
#                   model.append(Tr[index][0])
#             Index.append(model)     
# Index = pd.DataFrame(Index) 
# pd.DataFrame(Index).to_csv(f"./LSTM/3/0225tt/(IndexAll).csv",index=False, header = header)

# df = ImportCSV(f"./LSTM/3/0225tt/(IndexAll)",None)
# df = df.sort_values(by=['CC'], ascending=False)
# for model in df["model"][:20]:
#       print(model)
import numpy as np
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
trainOrtest = Tr
TPlus = 1
timeList=[3,3,3,3,3,6,3]
maxtime = max(timeList) #最大步長時間
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
trainOrtest = sc.fit_transform(trainOrtest)

x = []   #預測點的前 N 天的資料
y = []   #預測點
for t in range(maxtime, len(trainOrtest)-TPlus):  # 1258 是訓練集總數
      temp = np.hstack((trainOrtest[t-timeList[0]:t+1,0],trainOrtest[t-timeList[1]:t+1,1],
                 trainOrtest[t-timeList[2]:t+1,2],trainOrtest[t-timeList[3]:t+1,3],
                 trainOrtest[t-timeList[4]:t+1,4],trainOrtest[t-timeList[5]:t+1,5],
                 trainOrtest[t-timeList[6]:t+1,6])) # T-Tstep ~ T
      x.append(temp)
      y.append(trainOrtest[t+TPlus,-1]) # T+N
x, y = np.array(x), np.array(y)