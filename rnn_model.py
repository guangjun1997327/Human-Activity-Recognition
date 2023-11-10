import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def rnn(inputshape,num_lstm, num_layer,dense,droprate):
  model = keras.Sequential()
  model.add(layers.Input(shape=inputshape))
  for i in range(num_layer):
      model.add(layers.LSTM(num_lstm, dropout=droprate,return_sequences=True))
  model.add(layers.Dense(dense, activation = 'relu'))
  model.add(layers.Dense(12, activation = 'softmax'))
  return model

