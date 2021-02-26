import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore") # 경고 메시지를 무시
# print(train.tail())
# print(submission.tail())

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') ##1일치당겨오기
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') #2일치당겨오기 
        temp = temp.dropna()
        return temp.iloc[:-96] # 마지막2일치 빼고 전체 당겨오기

    elif is_train==False:
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]                     
        return temp.iloc[-48:, :] #마지막 하루치 데이터
#Target1ㅡ2 까지 합해서 칼럼 총 9개

#데이터 다시 x,y로 자르기 (train x,y자를거 미리 설정하기)
def split_xy(dataset, time_steps, y_row):
    x,y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_row
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-2] #총 9개 컬럼중 7일치 앞
        tmp_y = dataset[i:x_end_number, -2: ] #마지막 2일치 뒤로 빼기
        #(우리는 다음0~8일차중 y인7,8일차는 target만 구하면 되기 때문에)
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)    

#-------------------------------
# 1_1. train 데이터 불러오기

train = pd.read_csv('C:/data/태양광 발전량 예측data/train/train.csv', encoding='cp949')
df_train = preprocess_data(train, is_train=True) #디폴트값is_train=True
# print(df_train.shape) # (52464, 9)
dataset = df_train.to_numpy()
x = dataset.reshape(-1, 48, 9) #하루치 48번 칼럼 9개(1093, 48, 9)
# print(x.shape)
x,y = split_xy(dataset, 48, 1) #앞에서 설정한데로 x=7, y=2컬럼으로 됨
# print(x.shape) #(52416, 48, 7)
# print(y.shape) #(52416, 48, 2)

# 1_2. train데이터 불러오기
df_test = []  #x_pred
#for i in range(81):은 test.csv파일이 81개가 있는데 그거 한꺼번에 불러옴
for i in range(81):
    file_path = 'C:/data/태양광 발전량 예측data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) #def preprocess_data 위 설정참고
    df_test.append(temp)

df_test = pd.concat(df_test) #두 문자열을 하나의 문자열로 연결하여 반환
# print(x_test.shape) #(3888, 7)
# print(x_test.head(48))
test_dataset = df_test.to_numpy()
x_pred = test_dataset.reshape(81, 48, 7) #3888/48=81(2일치 나누기)
# print(x_pred.shape)

#=================================
# 1_3. 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                  train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, 
                                  train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    # (33545, 48, 7)
# print(x_test.shape)     # (10484, 48, 7)
# print(x_val.shape)      # (8387, 48, 7)
# print(y_train.shape)    # (33545, 48, 2)
# print(y_test.shape)     # (10484, 48, 2)
# print(y_val.shape)      # (8387, 48, 2)

#MinMaxScaler는 2차원만 가능하므로 데이터 합친후 다시 분배
x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1], x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1], x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1], x_pred.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)
x_val= scaler.transform(x_val)
x_pred= scaler.transform(x_pred)

x_train = x_train.reshape(33545, 48, 7)
x_test = x_test.reshape(10484, 48, 7)
x_val = x_val.reshape(8387, 48, 7)
x_pred = x_pred.reshape(81, 48, 7)

#y는 이미 shape있으므로 할 필요 없다.

##============================================
# 2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Reshape

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu',\
    input_shape=(48,7)))  #input (N, 48, 7) --x_train.shape[1], x_train.shape[2]
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Flatten())

# model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu')) # 48*2 = 96
model.add(Reshape((48, 2)))   #48,2 일치는 y를 뜻함
model.add(Dense(2, activation='relu'))

model.summary()

#============================================
#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = 'C:/data/modelCheckpoint/dacon_0119_{epoch:02d}-{val_loss:.4f}.hdf5'
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True ,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, 
                 validation_data=(x_val, y_val), callbacks=[es,cp,lr])

#=================================
#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)
y_pred = y_pred.reshape(3888, 2)

# 5. 데이터 저장 ----------이거 마저 하기
sub = pd.read_csv('C:/data/태양광 발전량 예측data/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_pred[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_pred[:,1]

sub.to_csv('C:\data\csv\submission0119_2.csv', index=False)

