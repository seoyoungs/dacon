##https://github.com/dongjaeseo/study/blob/main/practice/dacon_03_0120_01.py
###서동재님 깃허브
############ GHI빼고 기존 데이터로만

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv1D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean, maximum

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train=pd.read_csv('C:/data/dacon_data/train/train.csv', encoding='cp949')
submission = pd.read_csv('C:/data/dacon_data/sample_submission.csv', encoding='cp949')

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
x_train = df_train.to_numpy()
# df_train.iloc[:48] #원래 1094일까지 

df_test = []
for i in range(81):
    file_path = 'C:/data/dacon_data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
x_test = x_test.to_numpy()

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2] # target1,2 빼기
        tmp_y1 = data[x_end-1:x_end,-2] # targer1만
        tmp_y2 = data[x_end-1:x_end,-1] # 아예 뒤것만 target2
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

x,y1,y2 = split_xy(x_train,1)
print(x.shape,y1.shape,y2.shape) #(52464, 1, 7) (52464, 1) (52464, 1)

def split_x(data,timestep):
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))

x_test = split_x(x_test,1)

from sklearn.model_selection import train_test_split as tts  # 이렇게 패키지 부른 후 정의해도 된다.
x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x,y1,y2, train_size = 0.7,shuffle = True, random_state = 0)


def quantile_loss(q, y, pred):
    err=(y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

qunatile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#2. 모델링
def mymodel():
    model = Sequential()
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu',input_shape = (1,7)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(40, activation = 'relu'))
    # model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))
    return model

#3. 컴파일 훈련 (y1, y2 따로)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.35, verbose = 1)
epochs = 1000
bs = 32

# 내일!!
x = []
for i in qunatile_list:
    model = mymodel()
    filepath_cp = f'C:/data/modelCheckpoint/dacon_y1_quantile_{i:.1f}.hdf5'
    cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y1_val),callbacks = [es,cp,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

x = []
# 모레!!
for i in qunatile_list:
    model = mymodel()
    filepath_cp = f'C:/data/modelCheckpoint/dacon_y2_quantile_{i:.1f}.hdf5'
    cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y2_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y2_val),callbacks = [es,cp,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp2 = pd.concat(x, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2
        
submission.to_csv('C:/data/dacon_data/sub_0121_1.csv', index = False)
