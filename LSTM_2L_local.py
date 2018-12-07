import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
import mysql.connector
from keras.optimizers import Adam
from sklearn import preprocessing
import datetime
cnx = mysql.connector.connect(user='root', password='941012',
                              host='127.0.0.1',
                              database='test0')
cursor = cnx.cursor()
query = ("SELECT n1,n2,P1,T1,P2,T2 FROM test0.2016stablepoint LIMIT 1,25000")
cursor.execute(query)
result=cursor.fetchall()
x_list = np.array(result)
scaler = preprocessing.StandardScaler().fit(x_list)
train_x = scaler.transform(x_list)
#print train_x
query = ("SELECT T34 FROM test0.2016stablepoint LIMIT 1,25000")
cursor.execute(query)
result=cursor.fetchall()
y_list = np.array(result)
y_list = y_list-1300*np.ones((y_list.shape[0], 1))
#print y_list
#print np.hstack((train_x,y_list))
x_list = np.hstack((train_x,y_list))
result = []
sequence_length = 30
for index in range(len(x_list) - sequence_length + 1):
    result.append(x_list[index: index + sequence_length])
result = np.array(result)
#print result
row1 = int(round(0.8 * result.shape[0]))
row2 = int(round(0.9 * result.shape[0]))
row3 = int(round(0.2 * result.shape[0]))
train = result[:row1, :]
test = result[row2:, :]
np.random.seed(10)
np.random.shuffle(train)
X_train = train[:,:,0:6]
y_train = train[:,-1, -1]
np.random.seed(10)
np.random.shuffle(test)
X_test = test[:,:,0:6]
y_test = test[:,-1, -1]
y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
#print X_train
#print y_train
BATCH_INDEX = 0
model = Sequential()
# RNN cell
layers = [6, 80, 100, 1]
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

cursor.close()
cnx.close()

#for step in range(1001):
    # data shape = (batch_num, steps, inputs/outputs)
#    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+20, :, :]
#    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+20, :]
#    cost = model.train_on_batch(X_batch, Y_batch)
#    BATCH_INDEX += 20
#    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

#    if step % 500 == 0:
#        cost, mae = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
#        print('test cost: ', cost, 'test accuracy: ', mae)

from keras.callbacks import TensorBoard
starttime = datetime.datetime.now()
model.fit(
    X_train, y_train,
    batch_size=100, epochs=14,validation_split=0.1,verbose=0)
endtime = datetime.datetime.now()
print 'TIME:'+str((endtime - starttime).seconds)+'s'
predicted = model.predict(X_test)
predicted = predicted+1300*np.ones((predicted.shape[0], 1))
y_test = y_test+1300*np.ones((y_test.shape[0], 1))
#print X_test
#print y_test
predicted_pro = (np.mean(y_test)-np.mean(predicted))*np.ones(50)+predicted[0:50,0]
#print y_test[1:30]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test[0:50],label="Real")
ax.legend(loc='upper left')
plt.plot(predicted[0:50],label="Prediction",linestyle='--')
plt.xlabel('data point')
plt.ylabel('temperature (K)')
plt.legend(loc='upper left')
plt.ylim(1200,1450)
acc = np.ones((y_test.shape[0],1))-abs((predicted-y_test)/y_test)
acc = round(np.mean(acc),5)
plt.title('Accuracy:'+str(100*acc)+'%')
plt.show()
from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#model.save('/home/wuhang/models/model02.h5')

