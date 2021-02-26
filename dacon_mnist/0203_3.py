## y 값은 digit 지금 y 값이 주어지지 않음
## 따로 y값 나누는 거 아니다
### pca 안한게 10배는 잘나온다
# 베이스 라인 3등 참조

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, MaxPooling1D, Conv1D
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('C:/data/dacon_mnist/train.csv')
test = pd.read_csv('C:/data/dacon_mnist/test.csv')

# ==============  데이터 & 전처리  =========================
# print(train,test,sub)
# # distribution of label('digit') 
tra_di = train['digit'].value_counts()

# drop 인덱스
train2 = train.drop(['id','digit','letter'],1) # 인덱스 있는 3개 버리기
test2 = test.drop(['id','letter'],1) #인덱스 있는 것 버리기

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values
# print(train2.shape) #(2048, 784)
# print(test2.shape) # (20480, 784)

# 정규화(Minmax도 해보기) ---> standard보다 Minmax가 잘나온다
scaler = MinMaxScaler()
scaler.fit(train2)
scaler.transform(train2)
scaler.transform(test2)

# # reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)
# train2 = train2.reshape(-1,97,2,1)
# test2 = test2.reshape(-1,97,2,1) #4차원

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator() #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도

# ================== 모델링 ==============================
def modeling() :
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten()) #2차원
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10,activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍
    return model

re = ReduceLROnPlateau(patience=50, verbose=1, factor= 0.5)
ea = EarlyStopping(patience=100, verbose=1, mode='auto')
epochs = 1000
#KFlod대신 StratifiedKFold 써보기
#stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
result = 0
nth = 0
t_d = train['digit'] # y 값 부여

for train_index, valid_index in skf.split(train2, t_d):
    x_train = train2[train_index]
    x_valid = train2[valid_index]
    y_train = t_d[train_index]
    y_valid = t_d[valid_index]
    # print(x_train.shape, x_valid.shape) #(1946, 28, 28, 1), (102, 28, 28, 1)
    # print(y_train.shape, y_valid.shape) #(1946,) (102,)

    # 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습(fit_generator에서 할 것)
    train_generator = idg.flow(x_train,y_train,batch_size=8) #훈련데이터셋을 제공할 제네레이터를 지정
    valid_generator = idg2.flow(x_valid,y_valid) # validation_data에 넣을 것
    test_generator = idg2.flow(test2,shuffle=False)  # predict(x_test)와 같은 역할
    
    model = modeling()
    mc = ModelCheckpoint('C:/data/modelCheckpoint/0202_3_best_mc.h5', save_best_only=True, verbose=1)
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc']) # y의 acc가 목적
    img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=valid_generator, callbacks=[ea,mc,re])
    
    # predict
    # model.load_weights('C:/data/modelCheckpoint/0202_3_best_mc.h5')
    result += model.predict_generator(test_generator,verbose=True)/40 #a += b는 a= a+b
    # predict_generator 예측 결과는 클래스별 확률 벡터로 출력
    print('result:', result)

    # save val_loss
    hist = pd.DataFrame(img_fit.history)
    val_loss_min.append(hist['val_loss'].min())
    nth += 1
    print(nth, 'set complete!!') # n_splits 다 돌았는지 확인

# print(val_loss_min, np.mean(val_loss_min))
# model.summary()

#제출========================================
sub = pd.read_csv('C:/data/dacon_mnist/submission.csv')
sub['digit'] = result.argmax(1) # y값 index 2번째에 저장
sub
sub.to_csv('C:/data/dacon_mnist/answer/0202_5_mnist.csv',index=False)
