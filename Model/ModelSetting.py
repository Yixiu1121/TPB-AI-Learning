from tensorflow.keras import Sequential, losses, optimizers
from tensorflow.keras.layers import LSTM,Dropout,Dense,Flatten,RNN,SimpleRNN,RepeatVector,TimeDistributed,Input,Conv1D,MaxPooling1D,Bidirectional, Permute, Multiply
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn import svm
import numpy as np
from Model.Processing import dataFunction
from Control.ImportData import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import tensorflow as tf

# LayerNum = [16,16,8]

def deepLearning(name, para, gpara, LayerNum):
    # with tf.device('/GPU:0'):
        if name == "LSTM":
            model = Sequential()
            model.add(LSTM(LayerNum[0],input_shape= para.inputShape, return_sequences=True))
            for i in LayerNum[1:-1]:
                model.add(LSTM(i, return_sequences=True))
            model.add(LSTM(LayerNum[-1]))
            # model.add(Dropout(0.1))
            model.add(Dense(1, activation = gpara.activate))
            # opt = optimizers.RMSprop(lr=gpara.lr) ##調整learning rate

        elif name == "BiLSTM":
            model = Sequential()
            model.add(Bidirectional(LSTM(LayerNum[0], return_sequences=True, stateful=False), merge_mode="concat",input_shape= para.inputShape))
            for i in LayerNum[1:-1]:
                model.add(Bidirectional(LSTM(i, return_sequences=True)))
                # model.add((LSTM(i, return_sequences=True)))
            model.add(Bidirectional(LSTM(LayerNum[-1])))
            # model.add(Dropout(0.1))
            model.add(Dense(1, activation = gpara.activate))
            # opt = optimizers.RMSprop(lr=gpara.lr) ##調整learning rate    

        # elif name == "attentionBiLSTM":
        #     inputs = Input(input_shape= para.inputShape)
        #     #注意力層
        #     attentionProbs = Dense(para.inputShape[1], activation='softmax', name='attention_vec')(inputs)
        #     attentionMul = Multiply()([inputs, attentionProbs])
        #     attentionMul = Dense(64)(attentionMul)
        #     model = Sequential()
        #     model.add(LSTM(LayerNum[0],input_shape= para.inputShape, return_sequences=True))
        #     model.add(Dense(1,activation='sigmoid')(attentionMul))
        #     opt = optimizers.RMSprop(lr=gpara.lr)
            
        elif name == "RNN":
            model = Sequential()
            # model.add(Flatten())
            model.add(SimpleRNN(LayerNum[0], input_shape= para.inputShape, return_sequences=True))
            # for i in LayerNum[1:-1]:
            #     model.add(SimpleRNN(i, return_sequences=True))
            # model.add(RNN(16,activation='relu'))
            # model.add(Dense(LayerNum[-1], activation='relu'))
            model.add(Dense(1, kernel_initializer='normal', activation= gpara.activate))
            # opt = optimizers.RMSprop(lr=gpara.lr)
        elif name == "Seq2Seq":
            "多對一"
            inputs = Input(shape = para.inputShape)
            dropout_layer = Dropout(0)
            # add Dropout layer
            encoder_outputs = dropout_layer(inputs)
            encoder_outputs, state_h, state_c = LSTM(256, return_state=True)(encoder_outputs)
            encoder_states = [state_h, state_c]

            # Set up the decoder
            decoder_inputs = Input(shape= para.inputShape)
            decoder_embeddings = decoder_inputs  # Remove the embedding layer
            decoder_outputs, _, _ = LSTM(256, return_sequences=False, return_state=True)(decoder_embeddings, initial_state=encoder_states)
            outputs = Dense(1, activation="relu")(decoder_outputs) #多對多
            # outputs = Dense(1, activation="relu")(decoder_outputs)
            # Define the model
            model = Model([inputs, decoder_inputs], outputs)
        elif name == "Seq2Seq-R":
            "多對一"
            inputs = Input(shape = para.inputShape)
            dropout_layer = Dropout(0)
            # add Dropout layer
            encoder_outputs = dropout_layer(inputs)
            encoder_outputs, state_h, state_c = SimpleRNN(256, return_state=True)(encoder_outputs)
            encoder_states = [state_h, state_c]

            # Set up the decoder
            decoder_inputs = Input(shape= para.inputShape)
            decoder_embeddings = decoder_inputs  # Remove the embedding layer
            decoder_outputs, _, _ = SimpleRNN(256, return_sequences=False, return_state=True)(decoder_embeddings, initial_state=encoder_states)
            outputs = Dense(1, activation="relu")(decoder_outputs) #多對多
            # outputs = Dense(1, activation="relu")(decoder_outputs)
            # Define the model
            model = Model([inputs, decoder_inputs], outputs)
        elif name == "CNN-LSTM":
            cnn = Sequential()
            cnn.add(Conv1D(filters=64, kernel_size=1, activation="relu",input_shape= para.inputShape))
            # cnn.add(MaxPooling1D(pool_size=2))
            cnn.add(Flatten())
            model = Sequential()
            model.add(TimeDistributed(cnn))
            model.add(LSTM(16))
            model.add(Dense(1, activation = gpara.activate))
        opt = optimizers.RMSprop(lr=gpara.lr)
        model.compile(optimizer=opt, loss=gpara.loss, metrics=['mae'])
        model.summary()
        return model
def machineLearning(name, gpara):
    if name == 'SVM':
        model = svm.SVR(kernel=gpara.kernal, gamma=gpara.gamma ,epsilon=gpara.epsilon, C=gpara.C, degree=gpara.degree)
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
        # stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001, restore_best_weights=True) ## monitor = acc 
        history = model.fit(X_train, Y_train, epochs = gpara.epochs, batch_size = gpara.btcsz ,validation_split=0.2) #,callbacks = [stopping]
    return  history,model

def Prediction(model,name,X_test):
    if name == 'SVM':
        X_test = (X_test).reshape(len(X_test),-1)
    if name == 'Seq2Seq':
        X_test = [X_test, X_test]
    forcasting = model.predict(X_test)
    return forcasting

def ForcastCurve( modelname, plotRange, Npara, F_Inv, Y_Inv, GP, subtitle, title, fileName=""):
    

    x = np.arange(len(Y_Inv[:plotRange]))
    # 反正規
    YT = (Y_Inv).flatten()
    YP = (F_Inv).flatten()
    d = {  'Observation':YT, 'Forcast':YP }
    if modelname == "SVM":
        g = {
            'gamma':GP.gamma,
            'C':GP.C, 
            'epsilon':GP.epsilon,
            'degree':GP.degree, 
            'kernal':GP.kernal,
            'Tsteplist':f"{Npara.TStepList}",
        }
    else:
        g = {   'activate':GP.activate, 
                'opt':GP.opt,
                'epochs':GP.epochs,
                'batchSize':GP.btcsz,
                'loss':GP.loss,
                'Tsteplist':f"{Npara.TStepList}",
                'Layer':f"{Npara.Layer}",
                'Learning rate':GP.lr
                }
    # 合併觀測值&預測值+指標 字典
    
    res = {**d, **dataFunction.Index(YT,YP), **g} 
    # path = f"{Npara.ModelName}\{Npara.TStep}\{subtitle}"
    # CheckFile(path)
    pd.DataFrame(res).to_csv(f"{fileName}index.csv",index=False, header = True)

    plt.rcParams["figure.figsize"] = (8, 6)
    plt.figure()
    plt.plot(x,YT[:plotRange],label='Observation value')
    plt.plot(x,YP[:plotRange],label='Forcasting value')
    plt.axhline(2.2,color="red", linestyle="--")
    plt.title(f"{Npara.TPlus}All{title}Event")
    plt.xlabel("Time")
    plt.xlabel("WaterLevel")
    plt.legend()
    # plt.savefig('TestCase.png')
    return plt.savefig(f'{fileName}{title}Case.png')
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

def plotHistory(history,fileName):
    "loss curve"
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    return plt.savefig(f'{fileName}loss.png')

def plotEventMSF(dff, importPath, xlength, eventName):
    "Megi鬍鬚圖"
    
    plt.rcParams["font.family"] = 'Times New Roman'
    x = np.arange(xlength)
    
    
    plt.rcParams["figure.figsize"] = (11, 8)
    plt.figure()
    
    plt.plot(x,dff[1][:],color="cornflowerblue", label="Forecast")
    for p in range(2,len(dff.columns)):
        plt.plot(x[p-1:],dff[p][p-1:],color="cornflowerblue")
    plt.plot(x,dff[0],color="black", label="Observation") #, linestyle=":"
    plt.axhline(2.2,color="red", linestyle="--", label="Emergency level")
    ax = plt.gca()
    xmajorLocator = MultipleLocator(2) #設置間隔
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.spines['right'].set_visible(False) #邊框
    ax.spines['top'].set_visible(False)
    yminorLocator = MultipleLocator(.5/2) #将此y轴次刻度标签设置为0.1的倍数
    xminorLocator = MultipleLocator(1)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    parameters = {'axes.labelsize': 18,   #x, y 標籤字體大小
        'axes.titlesize': 20,     #标题的字体大小
          'figure.titlesize':20,
          'xtick.labelsize':16,
          'ytick.labelsize':16,
          'legend.fontsize':16} #
    plt.rcParams.update(parameters)
    plt.xlim(0, 65) #x軸從零開始
    plt.ylim(-2.0, 5.0)
    plt.title(f"Typhoon {eventName}")
    plt.xlabel("Time (h)", fontsize = 20)
    plt.ylabel("WaterLevel (m)", fontsize = 20)
    plt.legend(loc='upper right')
    return(plt.savefig(f"{importPath}{eventName}鬍鬚圖.png"))
    