import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import seaborn as sns
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("C:/Users/yixiu/.spyder-py3/TPB/TPB-AI-Learning")

from Control.ImportData import *

AD = ImportCSV("AD",None)
AD =AD[~AD['#1'].str.contains('#')]
AD.columns = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']
AD['Shihimen'] = pd.to_numeric(AD['Shihimen'], downcast='float')   #型別轉換
AD['WaterLevel'] = AD['WaterLevel']*100-AD['Tide']


corr = AD.corr()
fig.Figure(figsize=(15,15))
sns.set_theme(font='Times New Roman')
## annot 表示顯示方塊中數字
sns.heatmap(corr, square=True, annot=False, cmap="RdBu_r", xticklabels= True, fmt ='.2f')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Pic/correlation-difference.png')
plt.close()
print(AD.dtypes)
# ##繪製箱形圖
# sns.boxplot(y=AD['Tide'])
# print(AD['WaterLevel'])
# sns.boxplot(y=AD['WaterLevel'])
# plt.savefig('Pic/boxplot.png')

## Tide WL 的 RMSE(全部)
# Tide = AD['Tide']/100
# print(Tide[0])
# WL = AD['WaterLevel']
# RMSE = np.sqrt(mean_squared_error(WL, Tide)) 
# print(RMSE)

## Tide WL 的 RMSE(測試集)
# Ts = ImportCSV("Test",None)
# Ts =Ts[~Ts['#16'].str.contains('#')]
# Ts.columns = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']
# pred = Ts['Tide']/100
# print(pred[0])
# obs = Ts['WaterLevel']
# RMSE = np.sqrt(mean_squared_error(obs , pred)) 
# CC = ((obs - obs.mean())*(pred -pred.mean())).sum()/np.sqrt(((obs - obs.mean())**2).sum())/np.sqrt(((pred - pred.mean())**2).sum())
# print("RMSE",RMSE)
# print("CC",CC)