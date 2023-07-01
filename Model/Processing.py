import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error
from sklearn.metrics import r2_score
from Control.ImportData import *

class dataFunction():
    def SplitforOneRow(trainOrtest):
        """切分不同時間步長資料"""
        x = []   #預測點的前 N 天的資料
        y = []   #預測點
        for t in range(len(trainOrtest)-1):  # 1258 是訓練集總數
            x.append(trainOrtest[t,:]) # T-Tstep ~ T
            y.append(trainOrtest[t+1,-1]) # T+N
        x, y = np.array(x), np.array(y)  # 轉成numpy array的格式，以利輸入 RNN
        return x, y
    def Split(time,trainOrtest,TPlus,TStep):
        """切分相同時間步長資料"""
        x = []   #預測點的前 N 天的資料
        y = []   #預測點
        for t in range(time, len(trainOrtest)-TPlus):  # 1258 是訓練集總數
            x.append(trainOrtest[t-TStep:t+1,:]) # T-Tstep ~ T
            y.append(trainOrtest[t+TPlus,-1]) # T+N
        x, y = np.array(x), np.array(y)  # 轉成numpy array的格式，以利輸入 RNN
        return x, y       
    def SplitMuti(trainOrtest,timeList,endTimeList,TPlus):
        """切分多個不同時間步長資料"""
        maxtime = max(timeList) #最大步長時間
        x = []   #預測點的前 N 天的資料
        y = []   #預測點
        for t in range(maxtime, len(trainOrtest)-TPlus-max(endTimeList)):  # 1258 是訓練集總數
            arg_tuple = ()
            for i in range(len(timeList)):
                arg_tuple += tuple(trainOrtest[t-timeList[i]:t+1+endTimeList[i],i]) #取到
                 # 只取到t-1
            temp = np.hstack(arg_tuple) # T-Tstep ~ T....
            x.append(temp)
            y.append(trainOrtest[t+TPlus,-1]) # T+N
        x, y = np.array(x), np.array(y)  # 轉成numpy array的格式，以利輸入 RNN
        return x, y        
    def Reshape(FeatureN,x_2D):
        """二維轉成三維"""
        x_3D = np.reshape(x_2D, (x_2D.shape[0], x_2D.shape[1], FeatureN))
        return x_3D
    
    def Index(obs,pred):
        """ 訓練 & 測試 modelName T+N"""
        RMSE = np.sqrt(mean_squared_error(obs, pred))    #RMSE 極端值影響明顯
        MAE = mean_absolute_error(obs, pred)          #MAE        
        MSLE = mean_squared_error(obs, pred)           #MSLE 懲罰被低估的估計大於被高估的估計。
        NSE = (1-(np.sum((obs-pred)**2)/np.sum((obs-np.mean(obs))**2)))  #E接近1，表示模式质量好，模型可信度高；E接近0，表示模拟结果接近观测值的平均值水平，即总体结果可信，但过程模拟误差大；E远远小于0，则模型是不可信的。
        CE = r2_score(obs,pred)                      #  R2(CE)  是我們常用的效率係數CE 跟NSE一樣
        CC = ((obs - obs.mean())*(pred -pred.mean())).sum()/np.sqrt(((obs - obs.mean())**2).sum())/np.sqrt(((pred - pred.mean())**2).sum())
        d = {'RMSE':RMSE,'MAE':MAE,'CE':CE,'CC':CC, 'MSLE':MSLE}
        return d

class Processed_Data():
    def OneRowProcessing(self, Dset, Para):
        """一行"""
        self.para = Para
        self.trainingSet = (Dset.TrainingData)
        self.testSet = (Dset.TestData)
        self._Normalization()   
        return self._oneRowFeatureScaling()
    
    def DataProcessing(self, Dset, Para):
        """目前使用"""
        self.para = Para
        self.trainingSet = (Dset.TrainingData)
        self.testSet = (Dset.TestData)
        self._Normalization()   
        return self._FeatureScaling()

    def SameProcessing(self, Dset, Para):
        """時間步長相同"""
        self.para = Para
        self.trainingSet = (Dset.TrainingData)
        self.testSet = (Dset.TestData)
        self._Normalization()   
        return self._sameFeatureScaling()
    
    def _Normalization(self):
        """正規化訓練和測試"""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
        self.sc = MinMaxScaler(feature_range = (0, 1))
        # self.sc = MaxAbsScaler()
        self.training_set_scaled = self.sc.fit_transform(self.trainingSet)
        self.test_set_scaled = self.sc.transform(self.testSet)
        
        # DF2CSV(pd.DataFrame(self.test_set_scaled), "normal") #debug
        return self.sc
    
    def _oneRowFeatureScaling(self):
        """分成時間序列 X=t-n~t  Y=t+1  timeStep"""
        """Tplus 預測 T+n 時刻"""
        self.X_train,self.Y_train = dataFunction.SplitforOneRow(self.training_set_scaled)
        self.X_test,self.Y_test = dataFunction.SplitforOneRow(self.test_set_scaled)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0],1, self.X_train.shape[1]))   # 3*7 轉成 1*21
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0],1, self.X_test.shape[1]))
        return self.X_train,self.Y_train,self.X_test,self.Y_test
    
    def _FeatureScaling(self):
        """分成時間序列 X=t-n~t  Y=t+1  timeStep"""
        """Tplus 預測 T+n 時刻"""
        timeList = self.para.TStepList
        endTimeList = self.para.EDTStepList
        TPlus = self.para.TPlus
        Step = self.para.TStep
        FeatureN = self.para.FeatureN
        self.X_train,self.Y_train = dataFunction.SplitMuti(self.training_set_scaled,timeList,endTimeList,TPlus)
        self.X_test,self.Y_test = dataFunction.SplitMuti(self.test_set_scaled,timeList,endTimeList,TPlus)
        # DF2CSV(pd.DataFrame(self.X_test), "split") #debug
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0],1, self.X_train.shape[1]))   # 3*7 轉成 1*21
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0],1, self.X_test.shape[1]))
        
        return self.X_train,self.Y_train,self.X_test,self.Y_test
    
    def _sameFeatureScaling(self):
        """分成時間序列 X=t-n~t  Y=t+1  timeStep"""
        """Tplus 預測 T+n 時刻"""
        TPlus = self.para.TPlus
        Step = self.para.TStep
        # FeatureN = self.para.FeatureN
        self.X_train,self.Y_train = dataFunction.Split(Step,self.training_set_scaled,TPlus,Step)
        self.X_test,self.Y_test = dataFunction.Split(Step,self.test_set_scaled,TPlus,Step)
        # self.X_train = np.reshape(self.X_train, (self.X_train.shape[0],1, self.X_train.shape[1]*FeatureN))   # 3*7 轉成 1*21
        # self.X_test = np.reshape(self.X_test, (self.X_test.shape[0],1, self.X_test.shape[1]*FeatureN))
        return self.X_train,self.Y_train,self.X_test,self.Y_test

    def _InverseCol(self,y):
        y = y.copy()
        y -= self.sc.min_[-1]
        y /= self.sc.scale_[-1]
        return y

    # def _InverseCol(self,y):
    #     self.sc.fit_transform(np.reshape(self.trainingSet['WaterLevel'].to_frame(),(len(self.trainingSet['WaterLevel'].to_frame()),1)))
    #     y = self.sc.inverse_transform(np.reshape(y,(len(y),1)))
    #     return np.reshape(y,(len(y)))

    def _ForcastNormal(self, Fors):
        return self.sc.transform(Fors)

if __name__ == '__main__':
    path ='C:/Users/309/Documents/GitHub/TPB Code/Seq2Seq/best model'
    MP = '/0530-M/(1)'
    LP = '/0525-L/3128(5)'
    BP = '/0525-B/364(5)'
    SP = '/0525-364(3)/364(3)'
    new_dict = {'Model':[],'RMSE':[],'MAE':[],'CE':[],'CC':[], 'MSLE':[]}
    
    for i,name in zip([MP,LP,BP,SP],['SVM','LSTM','BiLSTM','Seq2Seq']):
        d = pd.read_csv(f'{path}{i}(train)index.csv')
        res = dataFunction.Index(d.Observation,d.Forcast)
        new_dict['Model'].append(name)
        new_dict['RMSE'].append(res['RMSE'])
        new_dict['MAE'].append(res['MAE'])
        new_dict['CC'].append(res['CC'])
        new_dict['CE'].append(res['CE'])
    pd.DataFrame(new_dict).to_csv(f"{path}/IndexAll_modify.csv",index=False)