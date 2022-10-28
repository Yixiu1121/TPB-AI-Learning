import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
class dataFunction():
    def Normalize(self,data):
        "正規化(0,1)"
        
        self.sc = MinMaxScaler(feature_range = (0, 1))
        data_set_scaled = self.sc.fit_transform(data)
        return data_set_scaled
    
    def Split(self,time,trainOrtest,TPlus,TStep):
        "切分不同時間步長資料"
        x = []   #預測點的前 N 天的資料
        y = []   #預測點
        for i in range(time, len(trainOrtest)-TPlus):  # 1258 是訓練集總數
            x.append(trainOrtest[i-TStep:i+1,:]) # T-Tstep ~ T
            y.append(trainOrtest[i+TPlus,-1]) # T+N
        x, y = np.array(x), np.array(y)  # 轉成numpy array的格式，以利輸入 RNN
        return x, y        
    def Reshape(FeatureN,x_2D):
        "二維轉成三維"
        x_3D = np.reshape(x_2D, (x_2D.shape[0], x_2D.shape[1], FeatureN))
        return x_3D
    def InverseCol(self,y):
        y = y.copy()
        y -= self.sc.min_[-1]
        y /= self.sc.scale_[-1]
        return y
    
    def Index(obs,pred):
        " 訓練 & 測試 modelName T+N"
        RMSE = np.sqrt(mean_squared_error(obs, pred))    #RMSE
        MAE = mean_absolute_error(obs, pred)          #MAE        
        CE = r2_score(obs,pred)                      #  R2(CE)  是我們常用的效率係數CE
        CC = ((obs - obs.mean())*(pred -pred.mean())).sum()/np.sqrt(((obs - obs.mean())**2).sum())/np.sqrt(((pred - pred.mean())**2).sum())
        d = {'RMSE':RMSE,'MAE':MAE,'CE':CE,'CC':CC}
        return d
