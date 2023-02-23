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
import numpy as np
D3 = [[12,2,3],
      [4,5,6],
      [7,8,9]]
D1 = np.reshape(D3, (1,9))
print(D1)

