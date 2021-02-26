#####  다시 해보자구~~~!!도전!!! >o< ^o^
### GHI, Td, Td-d
### 상관관계 후 WS 버리기 --> 상관성 젤 적음
#### 데이터를 더 쪼갤 수 있을까~~???

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten, Reshape, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor


# 파일 불러오기
train = pd.read_csv('C:/data/dacon_data/train/train.csv')
submission = pd.read_csv('C:/data/dacon_data/sample_submission.csv')

#1. DATA

# GHI, Td, T-Td column 추가 함수 생성
def Add_features(data):
    data.insert(1,'Hour_Minute',data['Hour'] * 2 + data['Minute'] // 30)
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    return data

# train data column정의 #### 리더보드 참고
def preprocess_data(data, is_train=True):
    data = Add_features(data)
    temp = data.copy()
    temp = temp[['Hour_Minute','TARGET','GHI','T-Td','DHI','DNI','RH', 'T']] # 상관관계 낮은 WS 삭제
          
    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train == False:
        temp = temp[['Hour_Minute','TARGET','GHI','T-Td','DHI','DNI','RH', 'T']]

        return temp.iloc[-48:, :] #뒤에서부터

df_train = preprocess_data(train)
print(df_train.shape)   # (52464, 10)
# print(df_train[:48])

###=====================================================================
# 시계열 데이터(함수) --> y1, y2 로 나눠 각각 훈련시키기
def split_xy(dataset, time_steps) :
    x, y1, y2 = [],[],[]
    for i in range(len(dataset)) :
        x_end = i + time_steps
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i:x_end, :-2] # ['Hour' ~ 'T']
        tmp_y1 = dataset[x_end-1:x_end,-2] # Target1
        tmp_y2 = dataset[x_end-1:x_end,-1]   # Target2
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

X = df_train.to_numpy()
# print(X.shape)      # (52464, 10)
x,y1,y2 = split_xy(X,1)
print(x.shape, y1.shape, y2.shape) #(52464, 1, 8) (52464, 1) (52464, 1)

#------------------------------------------test 함수정의
real_test = []
for i in range(81):
    file_path = 'C:/data/dacon_data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    real_test.append(temp)

real_test = pd.concat(real_test)
test = np.array(real_test)
print(real_test.shape) # (3888, 8)
x_test = real_test.to_numpy()

#####===================================================전처리
x_train, x_val, y1_train, y1_val = train_test_split(x,y1, train_size = 0.8,
                                        shuffle = False, random_state = 0)
x_train, x_val, y2_train, y2_val = train_test_split(x,y2,train_size = 0.8,
                                        shuffle = False, random_state = 0)

# print(x_train.shape,x_val.shape) #(36724, 1, 8) (15740, 1, 8)
print(y1_train.shape, y1_val.shape, y2_train.shape, y2_val.shape) #(41971, 1, 1) (10493, 1, 1) (41971, 1, 1) (10493, 1, 1)
x_train= x_train.reshape(41971*1, 8)
x_val= x_val.reshape(10493*1, 8)

# StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train= x_train.reshape(41971, 1, 8)
x_val= x_val.reshape(10493, 1, 8)
x_test= x_test.reshape(3888, 1, 8)
# print(x_train.shape,x_val.shape,x_test.shape) #(36724, 1, 8) (15740, 1, 8) (3888, 1, 8)

# 함수 : Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#2. 모델링 --> 이렇게 묶어서 하기(컴파일 할때 편함)

def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same',input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    return model

# 3. 컴파일, 훈련
#####========컴파일,  훈련, predict
#y1, y2 각각 저장하기
es = EarlyStopping(monitor = 'val_loss', patience = 13, mode='auto')
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 6, factor = 0.3, verbose = 1)
epochs = ep = 200
batch_size = ba = 32

### y1 내일
x = []
for i in q:
    model = modeling()
    filepath_cp = f'C:/data/modelCheckpoint/dacon_y1_0125_5_q_{i:.1f}.hdf5'
    # cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    # model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    # model.fit(x_train,y1_train,epochs = ep, batch_size = ba, validation_data = (x_val,y2_val),callbacks = [es,cp,lr])
    model = load_model(filepath_cp, compile=False)
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

x = []
### y2 모레
for i in q:
    model = modeling()
    filepath_cp = f'C:/data/modelCheckpoint/dacon_y2_0125_5_q_{i:.1f}.hdf5'
    # cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    # model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    # model.fit(x_train,y2_train,epochs = ep, batch_size = ba, validation_data = (x_val,y2_val),callbacks = [es,cp,lr])
    model = load_model(filepath_cp, compile=False)
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp2 = pd.concat(x, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2

##??????????????근데 이 지표가 잘됐다는거 평가하는 방법있을까?????????  로스밖에 없어
submission.to_csv('C:/data/dacon_data/sub_0125_5_2.csv', index = False)
