import matplotlib.pyplot as plt
import numpy as np
import mysql.connector
from sklearn import preprocessing
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
sequence_length = 20
for index in range(len(x_list) - sequence_length + 1):
    result.append(x_list[index: index + sequence_length])
result = np.array(result)
#print result
row1 = int(round(0.9 * result.shape[0]))
row2 = int(round(0.9 * result.shape[0]))
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
for i in range(0,2):
    a = np.array_split(y_train, 2, axis = 0)[i]
    #print 'train x '+str(i)
    #print a
    b = '/home/wuhang/LSTM_code/outdata'+str(i)+str(1)+'.npy'
    np.save(b,a)
    a = np.array_split(X_train, 2, axis = 0)[i]
    #print 'train y '+str(i)
    #print a
    b = '/home/wuhang/LSTM_code/inputdata'+str(i)+str(1)+'.npy'
    np.save(b,a)
np.save('/home/wuhang/LSTM_code/testinput.npy',X_test)
np.save('/home/wuhang/LSTM_code/testout.npy',y_test)
