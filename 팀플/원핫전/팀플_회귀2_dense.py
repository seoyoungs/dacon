import numpy as np
import db_connect as db
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 

query = "SELECT * FROM main_data_table"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())

# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('float32').to_numpy()
y_train = train_value.iloc[:,-1].astype('float32').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_test = test_value.iloc[:,1:-1].astype('float32').to_numpy()
y_test = test_value.iloc[:,-1].astype('float32').to_numpy()

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  
         train_size=0.8, random_state = 77, shuffle=True ) 


import timeit
start_time = timeit.default_timer()

# 2. 모델구성

model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 6)) 
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1)) 

# 3. 컴파일 훈련

modelpath = '../data/modelCheckpoint/regressor2.hdf5'
es= EarlyStopping(monitor='val_loss', patience=10)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_val,y_val), callbacks=[es,reduce_lr] )

# 4. 평가, 예측

loss, mae = model.evaluate(x_test, y_test, batch_size=16)
y_predict = model.predict(x_test)

print(loss)

# RMSE 
print("RMSE : ", RMSE(y_test, y_predict))

# R2 만드는 법
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))
