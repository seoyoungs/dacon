#https://github.com/skanwngud/skanwngud/blob/main/dacon/dacon_model_real.py
### 유주형님 깃허브 참고  ----> 속도 빠름

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean, maximum

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df_train=pd.read_csv('C:/data/dacon_data/train/train_0120.csv', header=0, index_col=0)
df_test=pd.read_csv('C:/data/dacon_data/test/test_0120.csv', header=0, index_col=0)
submission = pd.read_csv('C:/data/dacon_data/sample_submission.csv', encoding='cp949')

df_train = df_train.to_numpy()
df_test = df_test.to_numpy()

# print(df_train.shape) #(52560, 8)
# print(df_test.shape) #(27216, 6)

df_train = df_train.reshape(-1, 48, 8) #1일치 48번 (30분씩) (DHI/DNI/WS/RH/T/TARGET/target1/target2)
df_test = df_test.reshape(-1, 7, 48, 6) #7일치 각 48번씩 6일

def split_x(data, time_steps, y_col):
    x, y = list(), list()
    for i in range(len(data)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_col
        if y_end_number > (len(data)):
            break
        tmp_x = data[i : x_end_number, :]
        tmp_y = data[x_end_number : y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x,y = split_x(df_train, 7, 2) #x는 7일치 y는 2일치로 나누기

x=x[:, :, :, :6]  #4차원인데 앞에 인덱스0~6까지 쓰고
y=y[:, :, :, -2:] # y는 뒤에서 부터 두개 쓰고(전에 설정한 target1,2)

# print(x.shape) #(1087, 7, 48, 6)
# print(y.shape) #(1087, 2, 48, 2)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)

#MinMaxScaler를 위해 차원을 2차원으로 만들기
# print(x_train.shape) #(869, 7, 48, 6)
# print(x_test.shape) #(218, 7, 48, 6)

x_train = x_train.reshape(869, 7*48*6)
x_test = x_test.reshape(218, 7*48*6)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(869, 7, 48, 6)
x_test = x_test.reshape(218, 7, 48, 6)

qunatile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y, pred):
    err=(y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

for q in qunatile_list: 
    inputs = Input(shape=(7, 48, 6))
    dense = Conv2D(128, 2, padding='same', activation='relu')(inputs)
    dense = (MaxPooling2D(2))(dense)
    dense = Dense(units=64, activation='relu')(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Flatten()(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dense(96, activation='relu')(dense)
    dense = Reshape((2, 48, 1))(dense)
    outputs = Dense(1)(dense)

model = Model(inputs = inputs , outputs=outputs)

modelpath = 'C:\data\modelCheckpoint/dacon_0120_3_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(monitor='val_loss',filepath = modelpath,save_best_only=True)
es = EarlyStopping(monitor='val_loss',patience=10)
lr = ReduceLROnPlateau(monitor = 'val_loss',patience=5,factor=0.5)
model = Model(inputs = inputs , outputs=outputs)
model.compile(loss = lambda y,pred: quantile_loss(q,y,pred),metrics=['mae'] )
model.fit(x_train, y_train, epochs=20, batch_size=32, 
validation_split=0.3, callbacks=[es,cp,lr])

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("mae : ", result[1])

pred=model.predict(df_test)
# print("y_pred : ", y_pred)
pred=pred.reshape(81*2*48*1)
y_pred=pd.DataFrame(pred)
   
file_path='C:/data/dacon_data/quantile_loss_' + str(q) + '.csv'
y_pred.to_csv(file_path)

