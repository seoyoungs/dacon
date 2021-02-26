## y 값은 digit 지금 y 값이 주어지지 않음
## 따로 y값 나누는 거 아니다
### pca 안한게 10배는 잘나온다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *       # * 전부다
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
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

# pd.데이터 프레임 numpy로 변환
train2 = np.array(train2)
test2 = np.array(test2)
# print(train2.shape) #(2048, 784)
# print(test2.shape) # (20480, 784)

# 정규화(Minmax도 해보기) ---> standard보다 Minmax가 잘나온다
scaler = MinMaxScaler()
scaler.fit(train2)
scaler.transform(train2)
scaler.transform(test2)

# # reshape
train2 = train2.reshape(-1,14,14,4)
test2 = test2.reshape(-1,14,14,4)

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator() #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도

# ================== 모델링 ==============================
def modeling() :
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(14,14,4),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(64,(7,7),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    
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
    return model #digit가 0~9까지니까 output 10으로

re = ReduceLROnPlateau(patience=50, verbose=1, factor= 0.5)
ea = EarlyStopping(patience=100, verbose=1, mode='auto')
epochs = 1000
#KFlod대신 StratifiedKFold 써보기
#stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
val_accuracy_min = []
y_pred = 0
n = 0
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
    mc = ModelCheckpoint('C:/data/modelCheckpoint/0203_1_best_mc.h5', save_best_only=True, verbose=1)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc']) # y의 acc가 목적
    img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=valid_generator, callbacks=[ea,mc,re])
    # sparse_categorical_crossentropy : 범주형 교차 엔트로피와 동일하지만 
    # 이 경우 원-핫 인코딩이 된 상태일 필요없이 정수 인코딩 된 상태에서 수행 가능.

    # predict
    # model.load_weights('C:/data/modelCheckpoint/0203_1_best_mc.h5')
    y_pred += model.predict_generator(test_generator,verbose=1)/40 #a += b는 a= a+b
    # predict_generator 예측 결과는 클래스별 확률 벡터로 출력
    print('예측결과:', y_pred)

    # save val_loss
    hist = pd.DataFrame(img_fit.history) # 다시 데이터 프레임으로 변환
    val_loss_min.append(hist['val_loss'].min()) # 훈련 과정 시각화 (손실)
    val_accuracy_min.append(hist['accuracy'].min())
    print('loss 평균 : ', np.mean(val_loss_min))
    print('accuracy 평균 : ', np.mean(val_accuracy_min))
    n += 1
    print(n, 'set 완료!!') # n_splits 다 돌았는지 확인

model.summary()
#제출========================================
sub = pd.read_csv('C:/data/dacon_mnist/submission.csv')
sub['digit'] = y_pred.argmax(1) # y값 index 2번째에 저장
sub
sub.to_csv('C:/data/dacon_mnist/answer/0203_1_mnist.csv',index=False)

