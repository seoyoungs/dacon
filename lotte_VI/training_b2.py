import numpy as np
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

# ============== 데이터 load ===================
x = np.load("C:/data/npy/lotte_x_150.npy",allow_pickle=True)
y = np.load("C:/data/npy/lotte_y_150.npy",allow_pickle=True)
pred_x = np.load('C:/data/npy/lotte_pred_150.npy',allow_pickle=True)

print(x.shape, pred_x.shape, y.shape)  
#(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 128, 128, 3)
pred_x = preprocess_input(pred_x)

#  ================= 전처리 ====================
idg = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()
'''
# =============== train, vaild spilt ===========
x_train, x_valid, y_train, y_valid = train_test_split(
    x,y, train_size = 0.8, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,
                          batch_size=32, seed = 2048)
valid_generator = idg2.flow(x_valid,y_valid)

# ================ 모델링 ======================
eff2 = EfficientNetB4(include_top=False,weights='imagenet',
                         input_shape=x_train.shape[1:])
eff2.trainable = True
a = eff2.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(4048, activation= 'swish') (a)
a = Dropout(0.2) (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = eff2.input, outputs = a)

early_stopping = EarlyStopping(patience= 10)
lr = ReduceLROnPlateau(patience= 7, factor=0.5)
mc = ModelCheckpoint('C:/data/h5/lotte_255.h5',
                     save_best_only=True, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=50, steps_per_epoch= len(x_train) / 32,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])
'''
# predict
model = load_model('C:/data/h5/lotte_0326_1.h5')
# model.load_weights('C:/data/h5/lotte_0319_2.h5')
result = model.predict(pred_x,verbose=True)

print(result.shape)
sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/csv/lotte_150.csv',index=False)


'''
tta_steps = 30
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(pred_x,verbose=True, steps = len(x_valid) / 64)
    predictions.append(preds)

final_pred = np.mean(predictions, axis=0)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(final_pred,axis = 1)
sub.to_csv('C:/data/csv/lotte0320_1.csv',index=False)
'''

