import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error
from sklearn.metrics import r2_score

class dataFunction():
    def Split(time,trainOrtest,TPlus,TStep):
        "切分不同時間步長資料"
        x = []   #預測點的前 N 天的資料
        y = []   #預測點
        for t in range(time, len(trainOrtest)-TPlus):  # 1258 是訓練集總數
            x.append(trainOrtest[t-TStep:t+1,:]) # T-Tstep ~ T
            y.append(trainOrtest[t+TPlus,-1]) # T+N
        x, y = np.array(x), np.array(y)  # 轉成numpy array的格式，以利輸入 RNN
        return x, y        
    def Reshape(FeatureN,x_2D):
        "二維轉成三維"
        x_3D = np.reshape(x_2D, (x_2D.shape[0], x_2D.shape[1], FeatureN))
        return x_3D
    
    def Index(obs,pred):
        " 訓練 & 測試 modelName T+N"
        RMSE = np.sqrt(mean_squared_error(obs, pred))    #RMSE 極端值影響明顯
        MAE = mean_absolute_error(obs, pred)          #MAE        
        MSLE = mean_squared_error(obs, pred)           #MSLE 懲罰被低估的估計大於被高估的估計。
        CE = r2_score(obs,pred)                      #  R2(CE)  是我們常用的效率係數CE
        CC = ((obs - obs.mean())*(pred -pred.mean())).sum()/np.sqrt(((obs - obs.mean())**2).sum())/np.sqrt(((pred - pred.mean())**2).sum())
        d = {'RMSE':RMSE,'MAE':MAE,'CE':CE,'CC':CC}
        return d

class Processed_Data():
    def DataProcessing(self, Dset, Para):
        "步驟"
        self.para = Para
        self.trainingSet = (Dset.TrainingData)
        self.testSet = (Dset.TestData)
        self._Normalization()   
        return self._FeatureScaling()

    def _Normalization(self):
        "正規化訓練和測試"
        from sklearn.preprocessing import MinMaxScaler
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.training_set_scaled = self.sc.fit_transform(self.trainingSet)
        self.test_set_scaled = self.sc.transform(self.testSet)
        return self.sc

    def _FeatureScaling(self):
        "分成時間序列 X=t-n~t  Y=t+1  timeStep"
        "Tplus 預測 T+n 時刻"
        time = self.para.TStep
        TPlus = self.para.TPlus
        Step = self.para.TStep
        self.X_train,self.Y_train = dataFunction.Split(time,self.training_set_scaled,TPlus,Step)
        self.X_test,self.Y_test = dataFunction.Split(time,self.test_set_scaled,TPlus,Step)
        return self.X_train,self.Y_train,self.X_test,self.Y_test
    
    def _InverseCol(self,y):
        y = y.copy()
        y -= self.sc.min_[-1]
        y /= self.sc.scale_[-1]
        return y

    def _ForcastNormal(self, Fors):
        return self.sc.transform(Fors)

