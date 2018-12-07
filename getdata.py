from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
X_test = np.load('/home/wuhang/LSTM_code/testinput.npy')
y_test = np.load('/home/wuhang/LSTM_code/testout.npy')
model = models.load_model('/home/wuhang/LSTM_code/model01.h5')
predicted1 = model.predict(X_test)
#print predicted1
model = models.load_model('/home/wuhang/LSTM_code/model02.h5')
predicted2 = model.predict(X_test)
predicted = (predicted1 + predicted2)/2
predicted = predicted+1300*np.ones((predicted.shape[0], 1))
y_test = y_test+1300*np.ones((y_test.shape[0], 1))
acc = np.ones((y_test.shape[0],1))-abs((predicted-y_test)/y_test)
acc = 100*round(np.mean(acc),5)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test[1:50],label="Real")
ax.legend(loc='upper left')
plt.plot(predicted[1:50],label="Prediction")
plt.legend(loc='upper left')
plt.ylim(1100,1450)
plt.title('Accuracy:'+str(acc)+'%')
plt.show()

