from tensorflow.keras import Sequential, losses, optimizers
from tensorflow.keras.layers import LSTM,Dropout,Dense,Flatten,RNN,SimpleRNN,RepeatVector,TimeDistributed,Input,Conv1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn import svm
import numpy as np
from Model.Processing import dataFunction
from Control.ImportData import *
import matplotlib.pyplot as plt


# LayerNum = [16,16,8]
def deepLearning(name, para, gpara, LayerNum):
    if name == "LSTM":
        model = Sequential()
        model.add(LSTM(LayerNum[0], input_shape=(1,((para.TStep+1)*para.FeatureN)), return_sequences=True))
        for i in LayerNum[1:-1]:
            model.add(LSTM(i, return_sequences=True))
        # model.add(Dropout(0.3))
        model.add(LSTM(LayerNum[-1]))
        # model.add(Dropout(0.15))
        model.add(Dense(1, activation = gpara.activate))
        

    elif name == "RNN":
        model = Sequential()
        # model.add(Flatten())
        model.add(SimpleRNN(LayerNum[0], input_shape=( para.TStep+1, para.FeatureN)))
        # model.add(RNN(16,activation='relu'))
        model.add(Dense(LayerNum[-1], activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation= gpara.activate))
    
    elif name == "Seq2Seq":
        "多對一"
        inputs = Input(shape=(para.TStep+1, para.FeatureN))
        dropout_layer = Dropout(0.3)
        # add Dropout layer
        encoder_outputs = dropout_layer(inputs)
        encoder_outputs, state_h, state_c = LSTM(128, return_state=True)(encoder_outputs)
        encoder_states = [state_h, state_c]

        # Set up the decoder
        decoder_inputs = Input(shape=(para.TStep+1, para.FeatureN))
        decoder_embeddings = decoder_inputs  # Remove the embedding layer
        decoder_outputs, _, _ = LSTM(128, return_sequences=False, return_state=True)(decoder_embeddings, initial_state=encoder_states)
        outputs = Dense(1, activation="relu")(decoder_outputs)

        # Define the model
        model = Model([inputs, decoder_inputs], outputs)
    elif name == "CNN-LSTM":
        cnn = Sequential()
        cnn.add(Conv1D(filters=64, kernel_size=1, activation="relu", input_shape=( para.TStep+1, para.FeatureN)))
        cnn.add(MaxPooling1D(pool_size=2))
        cnn.add(Flatten())
        model = Sequential()
        model.add(TimeDistributed(cnn))
        model.add(LSTM(16))
        model.add(Dense(1, activation = gpara.activate))

    model.compile(optimizer=gpara.opt, loss=gpara.loss, metrics=['mae'])
    model.summary()
    return model
def machineLearning(name):
    if name == 'SVM':
        model = svm.SVR(kernel='rbf', gamma=0.125,epsilon=0.007813,C=8)
    return model

def FittingModel(model,name,X_train, Y_train, gpara):
    "注意:驗證集切分依順序"
    if name == 'SVM':
        X_train = (X_train).reshape(len(X_train),-1)
        history = model.fit(X_train, Y_train)
    elif name == "Seq2Seq":
        # stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001, restore_best_weights=True) ## monitor = acc
        # history = model.fit([X_train,X_train], Y_train, epochs = gpara.epochs, batch_size = gpara.btcsz ,validation_split=0.2,callbacks = [stopping]) 
        history = model.fit([X_train,X_train], Y_train, epochs = gpara.epochs, batch_size = gpara.btcsz ,validation_split=0.2)
    else:
        # model.compile(optimizer=self.gpara.opt, loss=self.gpara.loss, metrics=['mae'])
        # model.summary()
        stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001, restore_best_weights=True) ## monitor = acc 
        history = model.fit(X_train, Y_train, epochs = gpara.epochs, batch_size = gpara.btcsz ,validation_split=0.2,callbacks = [stopping])
    return  history,model

def Prediction(model,name,X_test):
    if name == 'SVM':
        X_test = (X_test).reshape(len(X_test),-1)
    if name == 'Seq2Seq':
        X_test = [X_test, X_test]
    forcasting = model.predict(X_test)
    return forcasting

def ForcastCurve( Npara, F_Inv, Y_Inv, GP,subtitle, fileName=""):
    x = np.arange(len(Y_Inv))
    # 反正規
    YT = (Y_Inv).flatten()
    YP = (F_Inv).flatten()
    d = {  'Observation':YT, 'Forcast':YP }
    if GP == "":
        g = {
            'Tstep':Npara.TStep
        }
    else:
        g = {   'activate':GP.activate, 
                'opt':GP.opt,
                'epochs':GP.epochs,
                'batchSize':GP.btcsz,
                'loss':GP.loss,
                'Tstep':Npara.TStep
                }
    # 合併觀測值&預測值+指標 字典
    
    res = {**d, **dataFunction.Index(YT,YP), **g} 
    path = f"{Npara.ModelName}\{Npara.TStep}\{subtitle}"
    CheckFile(path)
    pd.DataFrame(res).to_csv(f"{path}\{fileName}index.csv",index=False, header = True)
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.figure()
    plt.plot(x,YT,label='Observation value')
    plt.plot(x,YP,label='Forcasting value')
    plt.title(f"{Npara.TPlus}AllTestEvent")
    plt.xlabel("Time")
    plt.xlabel("WaterLevel")
    plt.legend()
    # plt.savefig('TestCase.png')
    return plt.savefig(f'{path}\{Npara.TStep}TestCase{fileName}.png')
def MSFForcastCurve(d, i, Y_Inv, F_Inv ):
    ""
    x = np.arange(len(Y_Inv))
    # 反正規
    YT = (Y_Inv).flatten()
    YP = (F_Inv).flatten()
    d[f'Observation{i}']= YT
    d[f'Forcast{i}']= YP 
    d["RMSE"].append(dataFunction.Index(YT,YP)["RMSE"])
    d["MAE"].append(dataFunction.Index(YT,YP)["MAE"])
    d["CE"].append(dataFunction.Index(YT,YP)["CE"])
    d["CC"].append(dataFunction.Index(YT,YP)["CC"])
    plotSeries(YT, YP, fileName = f"TestCase({i})")

def dictCSV(d, Npara, subtitle, fileName):
    path = f"{Npara.ModelName}\{Npara.TStep}\{subtitle}"
    CheckFile(path)
    pd.DataFrame(d).to_csv(f"{path}\({fileName}).csv",index=False, header = True)
    return path

def plotSeries(YT, YP, fileName):
    "t+1圖"
    x = np.arange(len(YT))
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.figure()
    plt.plot(x,YT,label='Observation value')
    plt.plot(x,YP,label='Forcasting value')
    plt.title(f"AllTestEvent")
    plt.xlabel("Time")
    plt.xlabel("WaterLevel")
    plt.legend()
    # plt.savefig('TestCase.png')
    return plt.savefig(f'{path}\{fileName}.png')

def plotHistory(history):
    "loss curve"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
def plotMegiMSF(dff, importPath):
    "Megi鬍鬚圖"
    x = np.arange(19)
    plt.rcParams["figure.figsize"] = (11, 8)
    plt.figure()
    plt.plot(x,dff[0],label='Observation value')
    plt.plot(x,dff[1][:],label='Forcasting value')
    for p in range(2,19):
        plt.plot(x[p-1:],dff[p][p-1:],label=f'Forcasting value{p}')
    plt.title(f"MegiEvent")
    plt.xlabel("Time")
    plt.xlabel("WaterLevel")
    plt.legend()
    return(plt.savefig(f"{importPath}Dujan鬍鬚圖.png"))

