
from abc import ABC, abstractmethod
from multiprocessing.dummy import Process
import numpy as np
from Control.ImportData import *
from tensorflow.keras import Sequential,losses,optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM,Dropout,Dense,Flatten,RNN,SimpleRNN
from tensorflow.keras.wrappers import scikit_learn
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import svm
from Model.Processing import dataFunction

class Para():
    ModelName:str
    TStep:int
    TPlus:int
    FeatureN:int
    ParamGrid:dict
    Scoring = "'neg_mean_absolute_error' 'neg_mean_squared_error' 'neg_median_absolute_error' 'r2'"
    PlotCaseN:int

class GPara():
    activate:str
    opt:str
    epochs:list
    btcsz:list
    loss:str

class DataSet(ABC):
    TrainingData = "訓練集"
    VaildationData = "驗證集"
    TestData = "測試集"

class Train(ABC):
    @abstractmethod
    def DataProcessing(self):
        "資料前處理"
    @abstractmethod
    def _ImportData(delf):
        "引入資料"
    @abstractmethod
    def _Normalization(self):
        "正規化"
    @abstractmethod
    def _FeatureScaling(self):
        "資料格式"
    @abstractmethod
    def ModelSetting(self):
        "建立模型"
    @abstractmethod
    def _LayerSetting(self):
        "層數"
    @abstractmethod
    def _FittingModel(self):
        "訓練模型"
    @abstractmethod
    def _Prediction(self):
        "預測"
    @abstractmethod
    def _GridSearching(self):
        "網格搜尋法"
    @abstractmethod
    def PlotResult(self):
        "視覺化結果"
    @abstractmethod
    def _LossFun(self):
        "畫訓練測試曲線"
    def _ForcastCurve(self):
        "預測歷線"

Process = dataFunction()
class L(Train):
    def Define(self,Para,Dset,GPara):
        'Para,Dset'
        self.para = Para
        self.gpara = GPara
        self.dataset = Dset
    
    def DataProcessing(self):
        "步驟"
        self._ImportData()
        self._Normalization()        
        self._FeatureScaling()

    def _ImportData(self):
        "取得訓練集 測試集"
        self.trainingSet = (self.dataset.TrainingData)
        self.testSet = (self.dataset.TestData)

    def _Normalization(self):
        "正規化訓練和測試"
        from sklearn.preprocessing import MinMaxScaler
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.training_set_scaled = self.sc.fit_transform(self.trainingSet)
        self.test_set_scaled = self.sc.transform(self.testSet)
        return self.training_set_scaled

    def _FeatureScaling(self):
        "分成時間序列 X=t-n~t  Y=t+1  timeStep"
        "Tplus 預測 T+n 時刻"
        time = self.para.TStep
        TPlus = self.para.TPlus
        TStep = self.para.TStep
        self.X_train,self.Y_train = Process.Split(time,self.training_set_scaled,TPlus,TStep)
        self.X_test,self.Y_test = Process.Split(time,self.test_set_scaled,TPlus,TStep)
        return 
    
    def ModelSetting(self,useGridSearch:bool):
        "是否使用網格搜尋法 模式名稱"
        if useGridSearch:
            self._LayerSetting(self.para.ModelName)
            self._GridSearching()
            self._Prediction()
        else:
            self._LayerSetting(self.para.ModelName)
            self._FittingModel()
            self._Prediction()

    def _LayerSetting(self,name):
        
        if name == "LSTM":
            model = Sequential()
            model.add(LSTM(16, input_shape=(self.para.TStep+1, self.para.FeatureN), return_sequences=True))
            model.add(LSTM(8, return_sequences=True))
            # model.add(LSTM(8, return_sequences=True)) 
            # model.add(Dropout(0.3))
            model.add(LSTM(8))
            # model.add(Dropout(0.15))
            model.add(Dense(1, activation=self.gpara.activate))  
            
        elif name == "RNN":
            model = Sequential()
            # model.add(Flatten())
            model.add(SimpleRNN(8,input_shape = (self.para.TStep+1, self.para.FeatureN)))
            # model.add(RNN(16,activation='relu'))
            model.add(Dense(16,activation='relu'))
            model.add(Dense(1,kernel_initializer='normal', activation=self.gpara.activate))
            # model.build(input_shape = (self.para.TStep, self.para.FeatureN))
            # model.summary()
        elif name == "SVM":
            model = svm.SVR(kernel='rbf', gamma=0.125,epsilon=0.007813,C=8)
        self.model = model
        return self.model

    def FittingModel(model,name,X_train, Y_train):
        "注意:驗證集切分依順序"
        if name == 'SVM':
            X_train = (X_train).reshape(len(X_train),-1)
            model.fit(X_train, Y_train)
        else:
            # model.compile(optimizer=self.gpara.opt, loss=self.gpara.loss, metrics=['mae'])
            # model.summary()
            stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001, restore_best_weights=True) ## monitor = acc 
            history = model.fit(X_train, Y_train, epochs = 200, batch_size = 16 ,validation_split=0.2,callbacks = [stopping])
        return  history, model

    def _Prediction(self):
        if self.para.ModelName == 'SVM':
            self.X_test = (self.X_test).reshape(len(self.X_test),-1)
        self.forcasting = self.model.predict(self.X_test)
        return self.forcasting    

    def _GridSearching(self):
        model = scikit_learn.KerasClassifier(build_fn=self.model,verbose = 0)
        grit = GridSearchCV(estimator=model, param_grid=self.para.ParamGrid, scoring = self.para.Scoring)
        grit_result = grit.fit(self.X_train, self.Y_train)
        print("Best: %f using %s" % (grit_result.best_score_, grit_result.best_params_))
        # for params, mean_score, scores in grid_result.grid_scores_:
        #     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        return grit

    def PlotResult(self,fileName):
        # if self.para.ModelName!='SVM':
        #     self._LossFun() 
        self._ForcastCurve(fileName)
        return "fileName"
    
    def _LossFun(self):
        print(self.history.history.keys()) #['loss', 'mae', 'val_loss', 'val_mae']
        # self.history.history['val_loss','loss','val_acc','acc']
        plt.plot(self.history.history['val_loss'],label='Val Loss')
        plt.plot(self.history.history['loss'],label='Loss')
        plt.legend()
        plt.savefig(f'{self.para.TPlus}LossCurve {self.para.ModelName}.png')
        return plt.close()

    def _InverseCol(self,y):
        y = y.copy()
        y -= self.sc.min_[-1]
        y /= self.sc.scale_[-1]
        return y

    def _ForcastCurve(self, fileName=""):
        x = np.arange(len(self.Y_test))
        # 反正規
        YT = (self._InverseCol(self.Y_test)).flatten()
        YP = (self._InverseCol(self.forcasting)).flatten()
        d = {   'Observation':YT,'Forcast':YP}
        g = {   'activate':self.gpara.activate, 
                'opt':self.gpara.opt,
                'epochs':self.gpara.opt,
                'batchSize':self.gpara.btcsz,
                'loss':self.gpara.loss,
                'Tstep':self.para.TStep
                }
        # 合併觀測值&預測值+指標 字典
        
        res = {**d, **Process.Index(YT,YP), **g} 
        path = f"{self.para.ModelName}\{self.para.TPlus}"
        CheckFile(path)
        pd.DataFrame(res).to_csv(f"{path}\{self.para.TStep}結果({fileName}).csv",index=False, header = True)
        plt.figure()
        plt.plot(x,YT,label='Observation value')
        plt.plot(x,YP,label='Forcasting value')
        plt.title(f"{self.para.TPlus}AllTestEvent")
        plt.xlabel("Time")
        plt.xlabel("WaterLevel")
        plt.legend()
        # plt.savefig('TestCase.png')
        return plt.savefig(f'{path}\{self.para.TStep}TestCase({fileName}).png')
    
#### 繪製單場
    def _PredictionCase(self):
        self.forcasting = self.model.predict(self.X_test)
        return self.forcasting

    # def msf(self,time,temp):
    #     "多步階預報"
    #     New_x, New_y = Process.Split(time,self.test_set_scaled,self.para.TPlus,self.para.TStep)
    #     f = self.forcasting
    #     # other_factor_forcast = [] #其他預測值
    #     for i in range(len(New_x)):
    #         New_x[i][-1][-1] = f[i]
    #         # New_x[i][-1][-2] = other_factor_forcast[i]
    #         if time>2 :
    #             New_x[i][-2][-1] = temp[-1][i]
    #             # New_x[i][-1][-2] = other_factor_forcast[i]
    #     print(time,f[0],temp[-1][0])
    #     print(New_x[0])

    #     self.X_test = New_x
    #     self.forcasting = self._Prediction()
    #     self.Y_test = New_y #畫圖用
    #     self._ForcastCurve('1029'+str(time)) #出圖&檔
    #     return self.forcasting

    # def MSF(self,t=6):
    #     "預設 t=6"
    #     Temp = [] #t+1
    #     for n in range(2,t+1):
    #         Temp.append(self._Prediction())
    #         self.msf(time = n, temp = Temp)

    def msf(self,time,temp):
        "多步階預報 t+2開始"
        new_x = np.delete(self.X_test,0, 1)     # 1:維度(由外到內) 移除3維第一行
        new_y = np.delete(self.Y_test, 0)   
        f = self.forcasting
        
        #觀測值 時間跟test一樣 #要正規化
        Shihimen = [] 
        Feitsui = []
        TPBRain = []
        SMOutflow = []
        FTOutflow = []
        Tide = []
        for i in range(len(f)):
            j = i+self.para.TStep + time -1 
            other_forcast = []  # 其他因子預報值
            other_forcast.append(Shihimen[j],Feitsui[j],TPBRain[j],SMOutflow[j],FTOutflow[j],Tide[j])
            other_forcast.append(f[i])  # 加入水位預測值
            new_x[i].append(other_forcast) #3維

        self.X_test = new_x
        self.forcasting = self._Prediction()
        self.Y_test = new_y #畫圖用
        self._ForcastCurve('1029'+str(time)) #出圖&檔
        return self.forcasting

    def MSF(self,t=6):
        "預設 t=6"
        Temp = [] #t+1
        for n in range(2,t+1):
            Temp.append(self._Prediction())
            self.msf(time = n, temp = Temp)



# LSTM.DataProcessing()
# LSTM.ModelSetting(False)

# 1. 放入預報值
# [[o1,o2,o3...]    t-2
#  [o1,o2,o3...]    t-1
#  [o1,o2,o3...]]   t

# [[o1,o2,o3...]    t-1
#  [o1,o2,o3...]    t
#  [f1,f2,f3...]]   t+1

# [[o1,o2,o3...]    t
#  [f1,f2,f3...]    t+1
#  [f1,f2,f3...]]   t+2

# [[f1,f2,f3...]    t+1
#  [f1,f2,f3...]    t+2
#  [f1,f2,f3...]]   t+3

# bug => MSF 
