import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 

'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC "


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()



# 원 핫으로 컬럼 추가해주는 코드!!!!!
df = pd.get_dummies(df, columns=["category", "dong"])
# 카테고리랑 동만 원핫으로 해준다 




# train, test 나누기

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)
# y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1],1)

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

kfold = KFold(n_splits=2, shuffle=True)

start_time = timeit.default_timer()

num = 0 
r2_list = []
rmse_list = []
loss_list = []

leaky_relu = tf.nn.leaky_relu

for train_index, test_index in kfold.split(x_train): 

    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]

    x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1], 1)
    x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1],1)

    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.8, random_state = 77, shuffle=False ) 

    # 2. 모델구성

    model = Sequential()
    model.add(LSTM(1024, activation='tanh', 
               input_shape= (6,1), unroll=False)) 
    model.add(Dense(512,activation='tanh'))
    model.add(Flatten())
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(1)) 

    # 3. 컴파일 훈련

    # modelpath = '../data/h5/one_hot_reg_1'+str(num)+'.hdf5'
    es= EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    # cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    cp = ModelCheckpoint('../data/h5/one_hot_reg_2_'+str(num)+'.hdf5', monitor='val_loss', save_best_only=True, verbose=1,mode='auto')
    model.compile(loss='mse', optimizer='rmsprop', metrics='mae')
    model.fit(x_train1, y_train1, epochs=25, batch_size=156, validation_data=(x_val,y_val), callbacks=[es,reduce_lr,cp] )

    # 4. 평가, 예측

    loss, mae = model.evaluate(x_test1, y_test1, batch_size=156)
    y_predict = model.predict(x_pred)

    print(loss)

    # RMSE 
    print("RMSE : ", RMSE(y_pred, y_predict))

    # R2 만드는 법
    r2 = r2_score(y_pred, y_predict)
    print("R2 : ", r2)

    num += 1

    r2_list.append(r2_score(y_pred, y_predict))
    rmse_list.append(RMSE(y_pred, y_predict))
    loss_list.append(loss)

print("LSTM 윈도우 없음")
print("r2 : ",r2_list)
print("RMSE : ",rmse_list)
print("loss : ",loss_list)

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

'''
leaky_relu
adadelta
RMSE :  0.09668588646694727
R2 :  0.7845471547295564

tanh
adam
RMSE :  0.015004378930168257
R2 :  0.9948112575549163
LSTM 윈도우 없음
r2 :  [0.9992032120682073, 0.9948112575549163]
RMSE :  [0.0058797429540654676, 0.015004378930168257]
loss :  [3.457082493696362e-05, 1.8847807950805873e-05]
6286.357138초 걸렸습니다.

tanh
rmsprop
r2 :  [0.9999155456155073, 0.9974719138580153]
RMSE :  [0.0019142474053310138, 0.0104732844885775]
loss :  [3.8070502341724932e-06, 8.74176748766331e-06]
6549.471142초 걸렸습니다.
'''



