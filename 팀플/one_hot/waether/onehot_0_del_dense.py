import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
import tensorflow.keras.backend as K
from torchvision.models import densenet121

# db 직접 불러오기 
fold_score_list = []
history_list=[]
csv_list = []

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def mish(x):
    return x * K.tanh(K.softplus(x))

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=10)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=5, factor=0.3, verbose=1)
    return er,mo,lr

def build_model(acti, opti, lr):   

    # 2. 모델구성
    
    inputs = Input(shape=(x_train.shape[1]),name='input')

    x = Dense(32,activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(16,activation="relu")(x)
    x = Dense(16,activation="relu")(x)

    outputs = Dense(1)(x)
    model = Model(inputs=inputs,outputs=outputs)

    # 3. 컴파일 훈련        
    model.compile(loss='mse', optimizer = opti(learning_rate=lr), metrics='mae')

    return model

def evaluate_list(model):
    score_list = []
    #4. 평가, 예측
    y_predict = model.predict(x_pred)

    # r2_list
    r2 = r2_score(y_pred, y_predict)
    score_list.append(r2)
    print('r2 : ', r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    score_list.append(rmse)
    print('rmse : ', rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    score_list.append(mae)
    print('mae : ', mae)
    # mse_list
    mse = mse_(y_pred, y_predict, squared=True)
    score_list.append(mse)
    print('mse : ', mse)

    return  score_list, y_predict

start_time = timeit.default_timer()


def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # pred = df.iloc[:,1:-1]

    if is_train == True:
        # train, test 나누기
        train_value = df[ '2020-09-01' > df['date'] ]
        x = train_value.iloc[:,1:-1].astype('float64')
        y = train_value.iloc[:,-1].astype('float64').to_numpy()
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1].astype('float64')
        y = test_value.iloc[:,-1].astype('float64').to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"]).to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y

x_pred, y_pred = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity,VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time", is_train = False)
x_train, y_train = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity,VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 

print(x_train.shape, y_train.shape) #(2213481, 46, 1) (2213481,)
print(x_pred.shape, y_pred.shape)   #(177408, 46, 1) (177408,)
print(x_val.shape, y_val.shape)     #(245943, 46, 1) (245943,) 

leaky_relu = tf.nn.leaky_relu
acti_list = [leaky_relu, mish, 'swish', 'elu', 'relu', 'selu','tanh']
opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad]
batch = 1000
lrr = 0.001
epo = 5000
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        modelpath = f'../data/h5/15_models_compare_w_{op_idx}_{ac_idx}.hdf5'
        model = build_model(acti, opti, lrr)
        # model = load_model(modelpath, custom_objects={'leaky_relu':tf.nn.leaky_relu, 'mish':mish})

        # 훈련
        er,mo,lr = callbacks(modelpath) 
        history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])
        # history_list.append(history)

        score, y_predict = evaluate_list(model)
        # 엑셀 추가 코드 
        # 경로 변경 필요!!!!
        df = pd.DataFrame(y_pred)
        df[f'{op_idx}_{ac_idx}'] = y_predict
        df.to_csv(f'../data/csv/15_models_compare_w_{op_idx}_{ac_idx}.csv',index=False)

        fold_score_list.append(score)
        print('=============parameter=================')
        print('optimizer : ', opti, '\n activation : ', acti, '\n batch_size : ', batch, '\n lr : ', lrr, '\n epochs : ', epo)
        print('r2   : ', score[0])
        print('rmse : ', score[1])
        print('mae : ', score[2])
        print('mse : ', score[3])
        print(f'======================================')


terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

print('=========================final score========================')
print("r2           rmse          mae            mse: ")
fold_score_list = np.array(fold_score_list).reshape(len(opti_list),len(acti_list),4)
print(fold_score_list)
'''
optimizer :  <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'> 
 activation :  tanh
 batch_size :  500
 lr :  0.001
 epochs :  5000
r2   :  0.858529394567721
rmse :  1.459906920570742
mae :  0.3961494602692392
mse :  2.131328216730347
'''


'''
r2           rmse          mae            mse:
[[[-5.92298676e+00  1.02126585e+01  5.49649047e+00  1.04298395e+02]
  [ 8.44900361e-01  1.52861267e+00  4.40564416e-01  2.33665670e+00]
  [ 4.01977482e-01  3.00158627e+00  7.74857640e-01  9.00952013e+00]
  [-1.41632750e+02  4.63555447e+01  3.89323034e+01  2.14883652e+03]
  [ 8.22257641e-01  1.63639269e+00  4.59273590e-01  2.67778104e+00]
  [-1.06818044e+03  1.26916347e+02  1.00507618e+02  1.61077592e+04]
  [ 7.96060049e-01  1.75284380e+00  5.14072264e-01  3.07246138e+00]]

 [[-2.65467904e+01  2.03717137e+01  1.14534569e+01  4.15006719e+02]
  [-8.89179189e-03  3.89865096e+00  9.26184704e-01  1.51994793e+01]
  [ 8.47649613e-01  1.51500422e+00  4.15438459e-01  2.29523778e+00]
  [-2.24276875e+02  5.82573020e+01  4.15590231e+01  3.39391324e+03]
  [-7.55987474e+01  3.39705748e+01  2.64951347e+01  1.15399995e+03]
  [-1.39225555e+00  6.00338017e+00  2.21482301e+00  3.60405735e+01]
  [-1.70050205e+01  1.64698207e+01  7.98561238e+00  2.71254994e+02]]

 [[ 8.46402441e-01  1.52119265e+00  4.55787772e-01  2.31402707e+00]
  [-2.52002459e+00  7.28223869e+00  4.51712012e+00  5.30310003e+01]
  [ 8.47836089e-01  1.51407676e+00  4.07593918e-01  2.29242843e+00]
  [ 8.41782261e-01  1.54390171e+00  4.39008375e-01  2.38363250e+00]
  [ 8.37787145e-01  1.56327252e+00  4.50968252e-01  2.44382099e+00]
  [-1.02749002e+01  1.30331206e+01  8.84278193e+00  1.69862233e+02]
  [-4.50336653e+01  2.63347888e+01  2.19764341e+01  6.93521099e+02]]

 [[ 6.51274255e-01  2.29210267e+00  7.16362785e-01  5.25373463e+00]
  [ 6.63467467e-01  2.25167439e+00  7.18844424e-01  5.07003756e+00]
  [ 6.62167291e-01  2.25601981e+00  7.94751774e-01  5.08962540e+00]
  [ 6.68748923e-01  2.23393592e+00  7.45197058e-01  4.99046969e+00]
  [ 6.45248390e-01  2.31182124e+00  7.16135007e-01  5.34451742e+00]
  [ 6.06117485e-01  2.43598950e+00  6.98671387e-01  5.93404485e+00]
  [ 6.02511508e-01  2.44711478e+00  7.66599791e-01  5.98837077e+00]]

 [[ 7.74350341e-01  1.84378128e+00  5.41400630e-01  3.39952942e+00]
  [ 9.01823743e-02  3.70227977e+00  7.43530335e-01  1.37068755e+01]
  [ 7.92894069e-01  1.76639704e+00  5.57881321e-01  3.12015852e+00]
  [ 8.26405818e-01  1.61718478e+00  4.91363358e-01  2.61528660e+00]
  [ 8.11588964e-01  1.68478788e+00  5.39466820e-01  2.83851020e+00]
  [ 8.06748008e-01  1.70629474e+00  5.82483169e-01  2.91144173e+00]
  [ 8.18237503e-01  1.65479501e+00  4.93357205e-01  2.73834651e+00]]

 [[ 6.25815369e-01  2.37429695e+00  7.81508755e-01  5.63728599e+00]
  [ 6.35709545e-01  2.34269611e+00  6.77566307e-01  5.48822508e+00]
  [ 6.25518551e-01  2.37523845e+00  7.83672463e-01  5.64175771e+00]
  [ 6.42238341e-01  2.32160836e+00  7.92430023e-01  5.38986539e+00]
  [ 6.10352064e-01  2.42285962e+00  7.20211029e-01  5.87024873e+00]
  [-3.33353942e-01  4.48192708e+00  2.06142149e+00  2.00876703e+01]
  [ 6.35772531e-01  2.34249358e+00  7.10819696e-01  5.48727617e+00]]]
'''



