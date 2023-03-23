# from Control.ImportData import *
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

# ## 取前20名
# df = ImportCSV(f"./LSTM/3/0225tt/(IndexAll)",None)
# df = df.sort_values(by=['CC'], ascending=False)
# for model in df["model"][:20]:
#       print(model)

from Control.ImportData import *
header = ["model","RMSE", "MAE", "CE", "CC", 'gamma', 'C', 'epsilon', 'degree', 'kernal', 'Tsteplist']
Index = []

for num in range(1,63):
      model = []
      Tr = ImportCSV(f"./SVM/6/SMO_FTO_WL_T/({num})index",None)
      model.append(f"({num})")
      for index in ["RMSE", "MAE", "CE", "CC", 'gamma', 'C', 'epsilon', 'degree', 'kernal', 'Tsteplist']:
            model.append(Tr[index][0])
      Index.append(model) 
Index = pd.DataFrame(Index)  
Index.columns = header  
Index = (Index).sort_values(by=['CC'], ascending=False) 
Index.to_csv(f"./SVM/6/SMO_FTO_WL_T/(IndexAll).csv",index=False, header = header)

## 取前20名
df = ImportCSV(f"./SVM/6/0321(1)/(IndexAll)",None)
df = df.sort_values(by=['CC'], ascending=False)
for model in df["model"][:20]:
      print(model)