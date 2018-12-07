import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
import mysql.connector
from keras.optimizers import Adam
from sklearn import preprocessing
sequence_length = 20
X_train = np.load('/home/wuhang/LSTM_code/inputdata11.npy')
y_train = np.load('/home/wuhang/LSTM_code/outdata11.npy')
#print len(X_train)
#print len(y_train)
BATCH_INDEX = 0
model = Sequential()
# RNN cell

layers = [5, 50, 100, 1]
model.add(LSTM(
        layers[1],
        input_shape=(sequence_length,layers[0]),
        return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
        layers[2],
        return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    layers[3]))
model.add(Activation("linear"))
    #model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
model.compile(loss="mse", optimizer="rmsprop", metrics=['mae', 'mape'])
from keras.callbacks import TensorBoard
start = datetime.datetime.now()
model.fit(
    X_train, y_train,
    batch_size=100, epochs=10,validation_split=0.05, verbose=0)
model.save('/home/wuhang/LSTM_code/model02.h5')
end = datetime.datetime.now()
print 'training complete, the time is '+str(end-start)

