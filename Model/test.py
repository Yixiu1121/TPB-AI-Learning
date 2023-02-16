# a = [[[0,1,2],
#       [3,4,5]],
#      [[0,1,2],
#       [3,4,5]]]
# print(a)
# LayerNum = [16,16,8]
# for i in LayerNum[1:-1]:
#       print(i)

# import os
# import tensorflow as tf
# # from tensorflow import keras

# new_model = tf.keras.models.load_model('../saved_model/LSTM163.h5')
# new_model.summary()

from tensorflow.keras import Sequential, losses, optimizers
from tensorflow.keras.layers import LSTM,Dropout,Dense,Flatten,RNN,SimpleRNN,RepeatVector,TimeDistributed,Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

d = {  'Observation':"YT", 'Forcast':'YP' }
g = {  'Observation':"YT", 'Forcast':'YP' }

res = {**d, **g} 