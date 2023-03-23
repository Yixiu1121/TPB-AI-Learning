from sklearn.feature_selection import VarianceThreshold
import numpy as np 
from Control.ImportData import *

init_input = ['Shihimen','Feitsui','TPB','SMInflow','SMOutflow','FTOutflow','Tide','WaterLevel']  #8個
input = ['Shihimen','Feitsui','TPB','SMOutflow','FTOutflow','Tide','WaterLevel']
Tr = ImportCSV("Train",None) #SM FT TPB SMInflow SMOutflow FTOutflow Tide WL
Ts =  ImportCSV("Test",None)
Tr = Tr[~Tr['#1'].str.contains('#')]
Ts = Ts[~Ts['#16'].str.contains('#')]
Tr.columns = init_input
Ts.columns = init_input
Tr = Tr[input]
Ts = Ts[input]

#閥值(threshold)為0.16,表示其變異程度低於0.16就會被removed
# variance_x=VarianceThreshold(threshold=0.16)
# variance_x.fit_transform(Tr) 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
#變成13*8features
for columnName in Tr.columns: 
    for i in range(12,0,-1):
        Tr[columnName+'t-'+str(i)] = Tr[columnName].shift(i)

for columnName in Ts.columns: 
    for i in range(12,0,-1):
        Ts[columnName+'t-'+str(i)] = Ts[columnName].shift(i)

x = Tr.drop(columns = ['WaterLevel'])
y=Tr['WaterLevel']
x_test = Ts.drop(columns = ['WaterLevel'])
y_test = Ts['WaterLevel']
x_test = x_test[12:]
y_test = y_test[12:]
x = x[12:]
y = y[12:]
# select_feature = ['Shihimen', 'Feitsui', 'TPB', 'SMOutflow', 'Feitsuit-7', 'SMOutflowt-6', 'Tidet-2', 'WaterLevelt-9', 'WaterLevelt-2', 'WaterLevelt-1']
select_feature = x.columns

# ['Shihimen', 'Feitsui', 'TPB', 'SMOutflow', 'Feitsuit-7', 'SMOutflowt-6', 'Tidet-2', 'WaterLevelt-9', 'WaterLevelt-2', 'WaterLevelt-1']
x = x[select_feature]
x_test = x_test[select_feature]
name=x.columns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
sc = MinMaxScaler(feature_range = (-1, 1))
training_set_scaled_x = sc.fit_transform(x)
x_test_set_scaled = sc.transform(x_test)
training_set_scaled_y = sc.fit_transform(np.reshape(y.to_numpy(),(len(y),1)))
y_test_set_scaled = sc.transform(np.reshape(y_test.to_numpy(),(len(y_test),1)))
X_train, X_vaild, y_train, y_vaild = train_test_split(training_set_scaled_x,training_set_scaled_y, test_size=0.02, shuffle=False)

regr = svm.SVR(kernel='rbf', gamma=0.125, epsilon=0.007813, C=8)
regr.fit(X_train, y_train)
# y_pre = regr.predict(x_test_set_scaled)
# plt.figure()
# xNum = np.arange(len(y_pre))
# plt.plot(xNum,y_test_set_scaled,label='Observation value')
# plt.plot(xNum,y_pre,label='Forcasting value')
# plt.axhline(2.2,color="red", linestyle="--")
# plt.title("t+12Event")
# plt.xlabel("Time")
# plt.xlabel("WaterLevel")
# plt.legend()
# plt.show()
## features select by PermutationImportance
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(regr, random_state=1).fit(X_train[:-11], y_train[11:])
eli5.show_weights(perm, feature_names = name.to_list())
from IPython.display import display, HTML
display(eli5.show_weights(perm, feature_names = name.to_list()).data)
## features select by SequentialFeatureSelector
sfs = SequentialFeatureSelector(regr, n_features_to_select=10, scoring='neg_mean_squared_log_error', cv=2)
sfs.fit(X_train, y_train)
sfs.get_support()
sfs.get_feature_names_out(input_features=name)



## 決策樹篩選
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
X_train, X_vaild, y_train, y_test = train_test_split(training_set_scaled_x,training_set_scaled_y, test_size=0.2, shuffle=False)
mae_values = []
train_mae_values = []
# 計算分數
X_train = pd.DataFrame(X_train)
X_vaild = pd.DataFrame(X_vaild)
X_train.columns = name
X_vaild.columns = name
for feature in x.columns:
    clf = DecisionTreeRegressor()
    clf.fit(X_train[feature].to_frame(), y_train)

    y_train_scored = clf.predict(X_train[feature].to_frame())
    train_mae_values.append(mean_absolute_error(y_train, y_train_scored))

    y_scored = clf.predict(X_vaild[feature].to_frame())
    mae_values.append(mean_absolute_error(y_test, y_scored))

# 建立Pandas Series 用於繪圖
mae_values = pd.Series(mae_values)
mae_values.index = x.columns

# 顯示結果
print(mae_values.sort_values(ascending=False))

train_mae_values = pd.Series(mae_values)
train_mae_values.index = x.columns

# 顯示結果
print(train_mae_values.sort_values(ascending=False))