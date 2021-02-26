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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor
import tensorflow.keras.backend as K

# 파일 불러오기
train = pd.read_csv('C:/data/dacon_data/train/train.csv')
submission = pd.read_csv('C:/data/dacon_data/sample_submission.csv')

#1. DATA

# GHI column 추가 함수
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

# train data column정의
def preprocess_data(data, is_train=True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train == False:
        temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

        return temp.iloc[-48:, :]

df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 10)
# print(df_train[:48])

#-------------test 함수정의
df_test = []
for i in range(81):
    file_path = 'C:/data/dacon_data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

df_test = pd.concat(df_test)
# print(df_test.shape) # (3888, 8)
x_test = df_test.to_numpy()

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
# print(x.shape, y1.shape, y2.shape) #(52464, 1, 8) (52464, 1) (52464, 1)

#####===========전처리
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x,y1,y2,
                           train_size = 0.7,shuffle = False, random_state = 0)

# print(x_train.shape,x_val.shape) #(36724, 1, 8) (15740, 1, 8)
# print(y1_train.shape, y1_val.shape, y2_train.shape, y2_val.shape) #(36724, 1) (15740, 1) (36724, 1) (15740, 1)
x_train= x_train.reshape(36724*1, 8)
x_val= x_val.reshape(15740*1, 8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train= x_train.reshape(36724, 1, 8)
x_val= x_val.reshape(15740, 1, 8)
x_test= x_test.reshape(3888, 1, 8)
# print(x_train.shape,x_val.shape,x_test.shape) #(36724, 1, 8) (15740, 1, 8) (3888, 1, 8)

# 함수 : Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#2. 모델링

def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    return model

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val loss', patience=5, mode='auto')
lr = ReduceLROnPlateau(monitor = 'val loss', patience=5, factor=0.4, verbose=1)
mc = '../data/modelcheckpoint/dacin_0121_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=mc, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
epochs =ep = 500
bs= batch_size=16

#####========컴파일,  훈련, predict
#y1, y2 각각 저장하기
x= []
for i in q :
    model = modeling()
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), 
    optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y1_train,epochs = ep, batch_size = bs, 
    validation_data = (x_val,y1_val),callbacks = [es,cp,lr])
    pred1 = model.predict(x_test).round(2)
    pred = pd.DataFrame(pred1)
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0 #0보다 작은것 다 0으로 바꾸기
n1_temp = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = n1_temp

x= []
for i in q :
    model = modeling()
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), 
    optimizer='adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train, y2_train ,epochs = ep, batch_size = bs, 
    validation_data = (x_val,y2_val),callbacks = [es,cp,lr])
    pred2 = model.predict(x_test).round(2)
    pred = pd.DataFrame(pred2)
    x.append(pred)
df_temp2 = pd.concat(x, axis=1)
df_temp2[df_temp2<0] = 0 #0보다 작은것 다 0으로 바꾸기
n2_temp = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = n2_temp

##??????????????근데 이 지표가 잘됐다는거 평가하는 방법있을까?????????
submission.to_csv('C:/data/dacon_data/sub_0122_1.csv', index = False)


