import numpy as np
import db_connect as db
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 


# 0 없다
'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

# 0 있다
query = "select * from main_data_table where dong = '0'"


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

# train, test 나누기

train_value = df[ '2020-08-31' >= df['date'] ]
train_value = train_value.iloc[:,1:].astype('int64').to_numpy()
# x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
# y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']
test_value = test_value.iloc[:,1:].astype('int64').to_numpy()

# x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
# y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def split_xy(dataset, time_steps, y_c):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end = i + time_steps
        y_end = x_end + y_c

        if y_end > len(dataset):
            break
        tmp_x = dataset[i:x_end, :-1]
        tmp_y = dataset[x_end:y_end, -1]
        x.append(tmp_x)
        y.append(tmp_y)
        # print(i)
    return np.array(x), np.array(y)

x_train, y_train = split_xy(train_value, 3696, 3696)
x_pred, y_pred = split_xy(test_value, 3696, 3696)
# print(x_train.shape)
# print(y_train.shape)
# print(x_train[0])
# print(y_train[0])


# kfold = KFold(n_splits=5, shuffle=True)

# num = 0 


# r2_list = []
# rmse_list = []
# loss_list = []


# for train_index, test_index in kfold.split(x_train): 

    # x_train1, x_test1 = x_train[train_index], x_train[test_index]
    # y_train1, y_test1 = y_train[train_index], y_train[test_index]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.8, random_state = 77, shuffle=True ) 

# 2. 모델구성
leaky_relu = tf.nn.leaky_relu
model = Sequential()
model.add(LSTM(512, input_shape = (3696,6), unroll=False)) 
model.add(Dense(256,activation=leaky_relu))
model.add(Dense(1)) 
model.summary()

# 3. 컴파일 훈련
modelpath = '../data/h5/regressor_LSTM.hdf5'
# es= EarlyStopping(monitor='val_loss', patience=10)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val,y_val), callbacks=[reduce_lr] )

# 4. 평가, 예측

loss, mae = model.evaluate(x_pred, y_pred, batch_size=32)
y_predict = model.predict(x_pred)

# RMSE 
print("RMSE : ", RMSE(y_pred, y_predict))

# R2 만드는 법
r2 = r2_score(y_pred, y_predict)
print("R2 : ", r2)



