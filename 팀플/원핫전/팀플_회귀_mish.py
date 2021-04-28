import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
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


# train, test 나누기
pred = df.iloc[:,1:-1]
train_value = df[ '2020-09-01' > df['date'] ]
x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']
x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

# print(x_train1.shape, y_train1.shape) #(3472128, 6) (3472128,)
# print(x_pred.shape, y_pred.shape)   #(177408, 6) (177408,)
#=======================================================================

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr


def build_model(acti, opti, lr):    
    #2. 모델
    inputs = Input(shape = (6,1),name = 'input')
    x = Conv1D(filters=1024,kernel_size=2,padding='same',strides = 2)(inputs)
    x = Activation(acti)(x)
    x = Conv1D(filters=1024,kernel_size=2,padding='same',strides = 2)(x)
    x = Activation(acti)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation(acti)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs,outputs=outputs)

    model.compile(loss = 'mse',optimizer = opti(learning_rate=lr), metrics = ['mae'])    

    return model

def evaluate_list(model):
    evaluate_list = []

    #4. 평가, 예측
    y_predict = model.predict(x_pred)

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

kfold = KFold(n_splits=2, shuffle=True)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
start_time = timeit.default_timer()
#==========================================================================
acti_list = ['elu', 'relu', 'selu','tanh','swish']
opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad, SGD]
acti = acti_list[2]
opti = opti_list[0]
batch = 156
lrr = 0.001
epo = 50

model = build_model(acti, opti, lrr)

#3. 훈련
modelpath ='../data/h5/regressor_mish.hdf5'
# er,mo,lr = callbacks(modelpath) 
# history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_valid, y_valid), callbacks = [er, lr, mo])
# #==========================================================================
# finish_time = timeit.default_timer()
# time = round(finish_time - start_time, 2)
# print('time : ', time)


# # list all data in history==============================
# import matplotlib.pyplot as plt
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['mae'])
# plt.plot(history.history['val_mae'])

# plt.title('mse & mae')
# plt.ylabel('epochs')
# plt.xlabel('loss, acc')
# plt.legend(['loss', 'val loss', 'mae', 'val mae'])
# plt.show()
# #=====================================================


# 모델로드
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'mish': mish})
model = load_model(modelpath, custom_objects={'mish':mish})

# 평가
print('=======================parameter====================')
# print('optimizer : ', opti, '\n activation : ', mish, '\n batch_size : ', batch, '\n lr : ', lrr, '\n epochs : ', epo)
evaluate = evaluate_list(model)

# time :  25268.1
# dict_keys(['loss', 'mae', 'val_loss', 'val_mae', 'lr'])
# =======================parameter====================
# optimizer :  <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>
#  activation :  relu
#  batch_size :  64
#  lr :  0.001
#  epochs :  50
# r2 :  0.7514047286674099
# rmse :  1.935256313538997
# mae :  0.6336414225017827
# mse :  3.7518646722363225
# 3_conv1d_5_relu_RMSprop.hdf5



