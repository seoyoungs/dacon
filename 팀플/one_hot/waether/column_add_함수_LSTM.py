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

# db 직접 불러오기 
fold_score_list = []
history_list=[]
csv_list = []

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def mish(x):
    return x * K.tanh(K.softplus(x))

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr

def build_model(acti, opti, lr):   

    # 2. 모델구성
    inputs = Input(shape = (x_train.shape[1],1),name = 'input')
    # 모델2
    x = LSTM(64, activation=acti)(inputs)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(32, activation=acti)(x)
    x = Dense(16, activation=acti)(x)
    x = Dense(8, activation=acti)(x)
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

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.8, random_state = 77, shuffle=True ) 

print(x_train.shape, y_train.shape) #(2213481, 46, 1) (2213481,)
print(x_pred.shape, y_pred.shape)   #(177408, 46, 1) (177408,)
print(x_val.shape, y_val.shape)     #(245943, 46, 1) (245943,) 

leaky_relu = tf.nn.leaky_relu
# tanh, RMSprop 안돌아감
acti_list = [leaky_relu, mish, 'swish', 'elu', 'relu', 'selu'] #   ,  ,  'tanh'
opti_list = [Adamax, Adagrad] # , ,RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad
batch = 1000
lrr = 1e-6
epo = 40
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        modelpath = f'../data/h5/21_weather_Adamax_LSTM_{op_idx}_{ac_idx}.hdf5'
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
        df.to_csv(f'../data/csv/21_weather_data_conv1d_{op_idx}_{ac_idx}.csv',index=False)

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
r2           rmse          mae            mse:				
[[[ 2.10039351e-01  3.44980693e+00  1.00834420e+00  1.19011678e+01]				
[ 2.38728745e-02  3.83482237e+00  9.40390918e-01  1.47058626e+01]				
[-2.39680211e-03  3.88608144e+00  1.21382222e+00  1.51016289e+01]				
[-7.03680749e-03  3.89506522e+00  1.27712114e+00  1.51715330e+01]				
[ 1.74661494e-02  3.84738655e+00  1.25168063e+00  1.48023832e+01]				
[ 1.73284276e-01  3.52915035e+00  1.10596290e+00  1.24549022e+01]]				
				
[[-4.19043440e-03  3.88955665e+00  1.29688070e+00  1.51286509e+01]				
[-7.35834559e-03  3.89568700e+00  1.33921746e+00  1.51763772e+01]				
[ 1.26940573e-02  3.85671846e+00  1.19824188e+00  1.48742773e+01]				
[ 1.40383178e-03  3.87870731e+00  1.32653640e+00  1.50443704e+01]				
[-3.38514718e-03  3.88799677e+00  1.33237286e+00  1.51165189e+01]				
[-1.71475488e-03  3.88475914e+00  1.21339386e+00  1.50913535e+01]]]				
				
r2           rmse          mae            mse:				
[[[ 2.19212301e-02  3.83865408e+00  1.22969346e+00  1.47352651e+01]				
[ 1.25674433e-03  3.87899296e+00  1.32844306e+00  1.50465864e+01]				
[-5.41627790e-03  3.89192997e+00  1.28507017e+00  1.51471189e+01]				
[-2.30270700e-03  3.88589904e+00  1.29986127e+00  1.51002113e+01]				
[ 7.98160255e-02  3.72331169e+00  9.94489796e-01  1.38630499e+01]				
[-9.11602969e-03  3.89908420e+00  1.36077764e+00  1.52028576e+01]]				
				
[[-6.81299752e-02  4.01147521e+00  1.20698735e+00  1.60919334e+01]				
[-4.18117949e-02  3.96174662e+00  8.21345729e-01  1.56954363e+01]				
[-3.02658990e-02  3.93973240e+00  6.78783086e-01  1.55214914e+01]				
[-4.39057611e-02  3.96572603e+00  9.95454000e-01  1.57269830e+01]				
[-6.94204856e-02  4.01389780e+00  8.87156229e-01  1.61113756e+01]				
[-2.46751246e-01  4.33393074e+00  2.20732860e+00  1.87829557e+01]]]	


  [-1.50485494e-02  3.91052862e+00  1.29627742e+00  1.52922341e+01]
  [ 6.70183125e-03  3.86840449e+00  1.10096708e+00  1.49645533e+01]
  [ 3.04584129e-03  3.87551709e+00  1.24746529e+00  1.50196327e+01]
  [-1.30428083e-02  3.90666310e+00  1.04843020e+00  1.52620166e+01]
  [-4.35090853e-02  3.96497249e+00  1.41045971e+00  1.57210069e+01]]

 [[-5.60356360e-02  3.98869977e+00  9.49470973e-01  1.59097259e+01]
  [-1.25366453e-01  4.11755154e+00  1.14605429e+00  1.69542307e+01]
  [-5.59037543e-02  3.98845070e+00  8.33755982e-01  1.59077390e+01]
  [-1.13518895e-01  4.09581995e+00  1.34736091e+00  1.67757410e+01]
  [-2.27018673e-02  3.92524334e+00  8.12158837e-01  1.54075353e+01]
  [-2.19797519e-02  3.92385732e+00  8.75944995e-01  1.53966563e+01]]]
'''





