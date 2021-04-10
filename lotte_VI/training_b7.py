import numpy as np
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2, EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101, VGG16
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

#데이터 지정 및 전처리
#data load
x = np.load("C:/data/npy/lotte_x_224.npy",allow_pickle=True)
y = np.load("C:/data/npy/lotte_y_224.npy",allow_pickle=True)
x_pred = np.load('C:/data/npy/lotte_pred_224.npy',allow_pickle=True)

# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

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

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=48, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = idg2.flow(x_pred, shuffle=False)

mc = ModelCheckpoint('C:/data/h5/lotte_0324_2.h5',save_best_only=True, verbose=1)
# efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
# mobile = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile = EfficientNetB2(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
# mobile = VGG16(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile.trainable = True
a = mobile.output
a = GlobalAveragePooling2D()(a)
a = Flatten()(a)
a = Dense(4048, activation= 'swish')(a)
a = Dropout(0.2)(a)
a = Dense(1000, activation= 'softmax')(a)

model = Model(inputs = mobile.input, outputs = a)

early_stopping = EarlyStopping(patience= 10)
lr = ReduceLROnPlateau(patience= 7, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=100, steps_per_epoch= len(x_train) / 48,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
# model = load_model('C:/data/h5/lotte_0323_1.h5')
model.load_weights('C:/data/h5/lotte_0324_2.h5')
result = model.predict(x_pred,verbose=True)

# 제출생성
sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/csv/lotte0323_2.csv',index=False)


# C:/data/h5/lotte_0323_1.h5 배치 48, 점수83
# lotte_0323_3.h5모델 lotte0323_2_1 80점

# predict
'''
model = load_model('C:/data/h5/lotte_0323_1.h5')
# model.load_weights('C:/data/h5/lotte_0323_1.h5')

idg3 = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1),
    zoom_range=0.1
     )
test_generator = idg3.flow(x_pred,shuffle=False)

from math import ceil
from tqdm import tqdm
preds_tta = []
tta_steps = 5
for i in tqdm(range(tta_steps)):
    test_generator.reset()
    preds = model.predict_generator(generator=test_generator, steps= ceil(len(x_pred)/48))
    preds_tta.append(preds)
    print(preds_tta)
preds_mean = np.mean(preds_tta, axis=0)
sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(preds_mean,axis = 1)
sub.to_csv('C:/data/csv/lotte0323_5.csv',index=False)
'''

