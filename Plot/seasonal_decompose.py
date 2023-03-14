from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import sys
sys.path.append("C:/Users/309/Documents/GitHub/TPB Code/TPB-AI-Learning")

from Control.ImportData import *
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
import matplotlib.pyplot as plt
AD = ImportCSV("Test",None)
AD =AD[~AD['#16'].str.contains('#')]
AD.columns = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']
# df = pd.DataFrame(sc.fit_transform(AD))
# df.columns = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']
# Decompose time series
#additive 特性的時間序列通常是代表有固定起伏、固定週期模式的線性時間序列。
result = seasonal_decompose(AD["WaterLevel"].values, model="additive", period=20) 
# result = seasonal_decompose(df["WaterLevel"].values, model="multiplicative", period=20)
result.plot()
plt.show()