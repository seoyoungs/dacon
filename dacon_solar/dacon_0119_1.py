###https://dacon.io/competitions/official/235680/codeshare/2033?page=1&dtype=recent&ptype=%20
##데이콘 참고 사이트 위 첨부
import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore") # 경고 메시지를 무시

train = pd.read_csv('C:/data/태양광 발전량 예측data/train/train.csv', encoding='cp949', header=0, index_col=0)
submission = pd.read_csv('C:/data/태양광 발전량 예측data/sample_submission.csv')

# print(train.tail())
# print(submission.tail())

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') ##1일치당겨오기
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') #2일치당겨오기 
        temp = temp.dropna()
        
        return temp.iloc[:-96] # 2일치 땡겨오기

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :] #뒤에 2일치 위에랑 총 다음날 1일치 예상


df_train = preprocess_data(train)
df_train.iloc[:48] #원래 1094일까지 있었는데 데이터 2일치 땡겨와서 1092일

# print(df_train.head())
# print(df_train.tail()) #Day 1092

train.iloc[48:96] #1일차
train.iloc[48+48:96+48] #2일차

df_test = []
#for i in range(81):은 test.csv파일이 81개가 있는데 그거 한꺼번에 불러옴
for i in range(81):
    file_path = 'C:/data/태양광 발전량 예측data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) #def preprocess_data 위 설정참고
    df_test.append(temp)

x_test = pd.concat(df_test) #두 문자열을 하나의 문자열로 연결하여 반환
# print(x_test.shape) #(3888, 7)
# print(x_test.head(48))

# print(df_train.head())
# print(df_train.iloc[-48:]) #끝에서부터 48개 출력

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=66)
x_train2, x_test2, y_train2, y_test2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=66)
# print(df_train.iloc[:, -2]) #target이랑 day만 자른다.
# df_train.iloc[:, :-2]  #Target1  Target2를 자른다.
# print(x_train_1.head())
# print(x_test.head())
# print(x_train2.shape, y_train2.shape) #(36724, 7) (36724,)


quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #백분위수 맞추기

# #2. 모델링
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM

# model = Sequential()
# model.add(Dense(52, input_dim=7))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(1))

#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
#Model은 함수형으로 입력하겠다는 뜻이다
from tensorflow.keras.layers import Dense, Input

#모델 1
input1= Input(shape=(7,))
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(5, activation='relu')(dense1)
#output1=Dense(3)(dense1)

#모델 2
input2= Input(shape=(7,))
dense2=Dense(10, activation='relu')(input2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
#output2=Dense(3)(dense2)

#모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
#from keras.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1,dense2]) #모델 1,2 병합
middle1=Dense(30)(merge1) #모델 합친후에도 layer추가 가능
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1) 

#데이터 합친 것 다시 나누기(100,3)2개니까 (100,6)된것 다시 나누기
#모델 분기1
output1= Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(1)(output1) #여기까지가 y1

#모델 분기2
output2= Dense(30)(middle1)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(1)(output2) #여기까지가 y2

#모델 선언 (함수형으로)
model = Model(inputs=[input1,input2], 
              outputs=[output1, output2]) #두개 이상은 list로 묶는다.
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', patience=6)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x_train1, x_train2], [y_train1, y_train2], epochs=10, validation_split=0.3,
          verbose=1, callbacks=[es])
'''
result=model.evaluate([x_test1,x_test2],[y_test1,y_test2], batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])
'''

submission.to_csv('C:\data\csv\submission0119_1.csv', index=True)


'''
결과
e_11_mae: 6.4167 - dense_15_mae: 6.7948 - val_loss: 295.2305 - val_dense_11_loss: 140.8573 
- val_dense_15_loss: 154.3736 - val_dense_11_mae: 6.1592 - val_dense_15_mae: 6.6501
'''



