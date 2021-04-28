import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D
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
query = "select * from main_data_table"


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

# train, test 나누기
train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

# print(x_train.shape)
'''
train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()
'''
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

start_time = timeit.default_timer()

kfold = KFold(n_splits=2, shuffle=True)

num = 0 


r2_list = []
rmse_list = []
loss_list = []


for train_index, test_index in kfold.split(x_train): 

    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]

    # x_train1=x_train1.reshape(x_train1.shape[0], x_train1.shape[1],1)
    # x_test1=x_test1.reshape(x_test1.shape[0], x_test1.shape[1],1)

    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.8, random_state = 77, shuffle=True ) 
    
    print(x_train1.shape)

    # 2. 모델구성
    leaky_relu = tf.nn.leaky_relu
    model = Sequential()
    model.add(Dense(2048, activation='tanh' ,input_dim= 6)) 
    model.add(Dense(512,activation='tanh'))
    # model.add(Dense(64,activation=leaky_relu))
    # model.add(Dense(32,activation=leaky_relu))
    # model.add(Dense(64,activation=leaky_relu))
    # model.add(Dense(64))
    # model.add(Dense(32))
    model.add(Dense(1)) 

    # 3. 컴파일 훈련
    mc = ModelCheckpoint('../data/h5/regressor6_'+str(num)+'.hdf5',save_best_only=True, verbose=1)
    es= EarlyStopping(monitor='val_loss', patience=8)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.002), metrics='mae')
    model.fit(x_train1, y_train1, epochs=100, batch_size=128, validation_data=(x_val,y_val), callbacks=[es,reduce_lr, mc] )

    # 4. 평가, 예측

    loss, mae = model.evaluate(x_test1, y_test1, batch_size=128)
    y_predict = model.predict(x_pred)

    # RMSE 
    print("RMSE : ", RMSE(y_pred, y_predict))

    # R2 만드는 법
    r2 = r2_score(y_pred, y_predict)
    print("R2 : ", r2)

    r2_list.append(r2_score(y_pred, y_predict))
    rmse_list.append(RMSE(y_pred, y_predict))
    loss_list.append(loss)

    num += 1


print("r2 : ",r2_list)
print("RMSE : ",rmse_list)
print("loss : ",loss_list)

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))


'''
RMSE :  2.0207088768506374
R2 :  0.7289662486895723
r2 :  [0.7373514345965776, 0.7391238762310719, 0.7349822557830787, 0.7571302089812757, 0.7289662486895723]
RMSE :  [1.9892051629012162, 1.9824818852779866, 1.9981566731855624, 1.912840730870225, 2.0207088768506374]
loss :  [3.4464402198791504, 3.527777671813965, 3.4165985584259033, 3.2635886669158936, 3.7043089866638184]
14713.299766초 걸렸습니다.
'''
