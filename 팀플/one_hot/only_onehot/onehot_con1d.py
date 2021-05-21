import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from tensorflow.keras.layers import LeakyReLU, PReLU
from time import time

fold_score_list = []
history_list=[]
time_list = []

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr


def build_model(acti, opti, lr):    
    #2. 모델
    inputs = Input(shape = (x_train.shape[1],1),name = 'input')
    x = Conv1D(filters=1024,kernel_size=2,padding='same',strides = 2, activation=acti)(inputs)
    x = Conv1D(filters=1024,kernel_size=2,padding='same',strides = 2, activation=acti)(x)
    x = Flatten()(x)
    x = Dense(512, activation=acti)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs,outputs=outputs)

    model.compile(loss = 'mse',optimizer = opti(learning_rate=lr), metrics = ['mae'])    

    return model

def evaluate_list(model):
    evaluate_list = []

    #4. 평가, 예측
    y_predict = model.predict(x_pred)

    print(y_predict.shape)
    print(y_pred.shape)

    # r2_list
    r2 = r2_score(y_pred, y_predict)
    evaluate_list.append(r2)
    print('r2 : ', r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    evaluate_list.append(rmse)
    print('rmse : ', rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    evaluate_list.append(mae)
    print('mae : ', mae)
    # mse_list
    mse = mse_(y_pred, y_predict, squared=True)
    evaluate_list.append(mse)
    print('mse : ', mse)

    return  evaluate_list

def mish(x):
    return x * K.tanh(K.softplus(x))

start_time = timeit.default_timer()
# db 직접 불러오기 =================================================

# 0 없다
# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"

# 0 있다
query = "SELECT * FROM `main_data_table`"
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
pred = df.iloc[:,1:-1]
train_value = df[ '2020-09-01' > df['date'] ]
x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']
x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()
#=======================================================================

fold = 3
kfold = KFold(n_splits=fold, shuffle=True)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(3472128, 42, 1) (3472128,)
print(x_pred.shape, y_pred.shape)   #(177408, 42, 1) (177408,)


acti_list = ['swish', 'elu', 'relu', 'selu','tanh']
opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad, SGD]
# acti = acti_list[4]
opti = opti_list[0]
# batch = 80
# lrr = 0.01
# epo = 1000
batch = 64
lrr = 0.001
epo = 200
for acti in acti_list:
    num = 0 
    for train_index, test_index in kfold.split(x_train):             

        x_train1, x_test1 = x_train[train_index], x_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]
        
        x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=True ) 


        model = build_model(acti, opti, lrr)

        #3. 훈련
        modelpath =f"./mitzy/data/hdf5/12_conv1d_encoding_{acti}_RMSprop.hdf5"
        er,mo,lr = callbacks(modelpath) 
        history = model.fit(x_train1, y_train1, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val, y_val), callbacks = [er, lr, mo])
        history_list.append(history)
        #==========================================================================
        # finish_time = timeit.default_timer()
        # time = round(finish_time - start_time, 2)
        # print('time : ', time)

        # 모델로드
        model = load_model(modelpath)

        # 평가
        score = evaluate_list(model)
        fold_score_list.append(score)
        print('=============parameter=================')
        print('optimizer : ', opti, '\n activation : ', acti, '\n batch_size : ', batch, '\n lr : ', lrr, '\n epochs : ', epo)
        print(f'============{num}fold=================')
        print('r2   : ', score[0])
        print('rmse : ', score[1])
        print('mae : ', score[2])
        print('mse : ', score[3])
        print(f'======================================')

        num += 1

    terminate_time = timeit.default_timer() # 종료 시간 체크  
    print("%f초 걸렸습니다." % (terminate_time - start_time))
    time_list.append(terminate_time - start_time)

print('=========================final score========================')
print("r2           rmse          mae            mse: ")
fold_score_list = np.array(fold_score_list).reshape(len(acti_list),fold,len(score))
print(fold_score_list)
print('activation별 time_list: ', time_list)

# list all data in history==============================
import matplotlib.pyplot as plt
print(history.history.keys())

fig = plt.figure(figsize=(100,30))

for i in range(len(acti_list)*fold):
    n = int(i/3)
    ax = fig.add_subplot(len(acti_list),fold,i+1)
    ax.plot(history_list[i].history["loss"])
    ax.plot(history_list[i].history["val_loss"])
    ax.plot(history_list[i].history["mae"])
    ax.plot(history_list[i].history["val_mae"])
    ax.set_title(f'{acti_list[n]}')
    ax.legend(['loss', 'val loss', 'mae', 'val mae'])
fig.tight_layout(h_pad = 8, w_pad = 8)
plt.show()
#=====================================================



""" # list all data in history==============================
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('mse & mae')
plt.ylabel('epochs')
plt.xlabel('loss, acc')
plt.legend(['loss', 'val loss', 'mae', 'val mae'])
plt.show()
#===================================================== """




# time :  24216.98
# dict_keys(['loss', 'mae', 'val_loss', 'val_mae', 'lr'])
# =======================parameter====================
# optimizer :  <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>
#  activation :  elu
#  batch_size :  64
#  lr :  0.001
#  epochs :  50
# (177408, 1)
# (177408,)
# r2 :  0.7548393615893347
# rmse :  1.9218409061552386
# mae :  0.6164727594619778
# mse :  3.6934724685715885
# 9_conv1d_4_elu_RMSprop.hdf5


""" 
=============parameter=================
optimizer :  <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>
 activation :  swish
 batch_size :  64
 lr :  0.001
 epochs :  200
============0fold=================
r2   :  0.9999999876061144
rmse :  2.3189463887072063e-05
mae :  5.068597430229394e-06
mse :  5.377512353698194e-10
====================================== 
"""





