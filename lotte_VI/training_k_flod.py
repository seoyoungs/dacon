# https://www.kaggle.com/phsaikiran/the-simpsons-characters-classification
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7, EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

#데이터 지정 및 전처리
x = np.load("C:/data/npy/lotte_x.npy",allow_pickle=True)
x_pred = np.load('C:/data/npy/lotte_pred.npy',allow_pickle=True)
y = np.load("C:/data/npy/lotte_y.npy",allow_pickle=True)
# y1 = np.zeros((len(y), len(y.unique())))
# for i, digit in enumerate(y):
#     y1[i, digit] = 1

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 

idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

y = np.argmax(y, axis=1)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet, EfficientNetB4, EfficientNetB2

'''
efficientnetb7 = EfficientNetB7(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
efficientnetb7.trainable = False
a = efficientnetb7.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(64) (a)
a = BatchNormalization() (a)
a = Activation('relu') (a)
a = Dense(128) (a)
a = BatchNormalization() (a)
a = Activation('relu') (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnetb7.input, outputs = a)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-5,epsilon=None),
                metrics=['acc'])
'''

# ================== 모델링 ==============================
seed = 0 # 0 일때 난수값 되도록 설정
np.random.seed(seed)
tf.random.set_seed(3)
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

i = 0
val_loss_min = []
val_acc_max = []

for train, valid in skf.split(x,y):
    i += 1
    x_train = x[train]
    x_valid = x[valid]
    y_train = y[train]
    y_valid = y[valid]

    train_generator = idg.flow(x_train,y_train,batch_size=128)
    # seed => random_state
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = x_pred

    mobile = EfficientNetB2(include_top=False,weights='imagenet',
                             input_shape=x_train.shape[1:])
    mobile.trainable = True
    a = mobile.output
    a = GlobalAveragePooling2D() (a)
    a = Flatten() (a)
    a = Dense(4048, activation= 'swish') (a)
    a = Dropout(0.2) (a)
    a = Dense(1000, activation= 'softmax') (a)

    model = Model(inputs = mobile.input, outputs = a)

    #3. Compile, Train
    mc = ModelCheckpoint('C:/data/h5/lotte_0319_1_'+str(i)+'.h5',save_best_only=True, verbose=1)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stopping = EarlyStopping(patience= 25)
    lr = ReduceLROnPlateau(patience= 15, factor=0.5)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.15), 
                    loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 128,
        validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    val_acc_max.append(hist['val_accuracy'].max())

    print("val_loss_min :", np.mean(val_loss_min))
    print("val_acc_max :", np.mean(val_acc_max))

'''
# predict
model = load_model('C:/data/h5/lotte_0319_2.h5')
# model.load_weights('C:/data/h5/lotte_0318_4.h5')
result = model.predict(x_pred,verbose=True)

tta_steps = 20
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(x_pred,verbose=True, steps = len(valid) / 128)
    predictions.append(preds)

final_pred = np.mean(predictions, axis=0)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(final_pred,axis = 1)
sub.to_csv('C:/data/csv/lotte0319_2_2.csv',index=False)


########## 이거로 해보기 #########
# predict
model = load_model('C:/data/h5/lotte_0319_2.h5')
# model.load_weights('C:/data/h5/lotte_0318_4.h5')
result = model.predict(x_pred,verbose=True)

tta_steps = 15
predictions = []

for i in tqdm(range(tta_steps)):
    preds = Model.predict_generator(x_pred, steps = len(x_valid)/128, verbose=True)
    predictions.append(preds)

final_pred = np.mean(predictions, axis=0)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.mean(np.equal(np.argmax(y_valid, axis=-1), np.argmax(final_pred, axis=-1)))
sub.to_csv('C:/data/csv/lotte0320_2.csv',index=False)
'''


