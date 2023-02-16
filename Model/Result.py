def _FittingModel(self):
        "注意:驗證集切分依順序"
        if self.para.ModelName == 'SVM':
            self.X_train = (self.X_train).reshape(len(self.X_train),-1)
            self.model.fit(self.X_train, self.Y_train)
        else:
            self.model.compile(optimizer=self.gpara.opt, loss=self.gpara.loss, metrics=['mae'])
            self.model.summary()
            stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001, restore_best_weights=True) ## monitor = acc 
            self.history = self.model.fit(self.X_train, self.Y_train, epochs = self.gpara.epochs, batch_size = self.gpara.btcsz,validation_split=0.2,callbacks = [stopping])
        return  self.model

def _Prediction(self):
    if self.para.ModelName == 'SVM':
        self.X_test = (self.X_test).reshape(len(self.X_test),-1)
    self.forcasting = self.model.predict(self.X_test)
    return self.forcasting
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
