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

df = ImportCSV(f"./LSTM/3/0225tt/(IndexAll)",None)
df = df.sort_values(by=['CC'], ascending=False)
for model in df["model"][:20]:
      print(model)