import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import PIL.Image as pilimg
import tensorflow as tf

######################################################
# File Load
train = pd.read_csv("C:/data/dacon_mnist2/dirty_mnist_answer.csv")
print(train.shape)  # (50000, 27)

sub = pd.read_csv("C:/data/dacon_mnist2/sample_submission.csv")
print(sub.shape)    # (5000, 27)

######################################################

#1. DATA

#### train
df_x = []

for i in range(0,50000):
    if i < 10 :
        file_path = 'C:/data/dacon_mnist2/dirty_mnist/0000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = 'C:/data/dacon_mnist2/dirty_mnist//000' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = 'C:/data/dacon_mnist2/dirty_mnist/00' + str(i) + '.png'
    elif i >= 1000 and i < 10000 :
        file_path = 'C:/data/dacon_mnist2/dirty_mnist/0' + str(i) + '.png'
    else : 
        file_path = 'C:/data/dacon_mnist2/dirty_mnist/' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_x.append(pix)

x = pd.concat(df_x)
x = x.values
print("x.shape ", x.shape)       # (12800000, 256) >>> (50000, 256, 256, 1)
print(x[0,:])
x = x.reshape(50000, 256, 256, 1)
x = x/255
x = x.astype('float32')

y = train.iloc[:,1:]
print("y.shape ", y.shape)    # (50000, 26)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=47)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ImageGenerator >> 그림 확인하기
idg = ImageDataGenerator(
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2
    )
idg2 = ImageDataGenerator()

#### pred
df_pred = []

for i in range(0,5000):
    if i < 10 :
        file_path = 'C:/data/dacon_mnist2/test_dirty_mnist/5000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = 'C:/data/dacon_mnist2/test_dirty_mnist/500' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = 'C:/data/dacon_mnist2/test_dirty_mnist/50' + str(i) + '.png'
    else : 
        file_path = 'C:/data/dacon_mnist2/test_dirty_mnist/5' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pred.append(pix)

x_pred = pd.concat(df_pred)
x_pred = x_pred.values
print(x_pred.shape)       # (1280000, 256) >>> (5000, 256, 256, 1)
x_pred = x_pred.reshape(5000, 256, 256, 1)
x_pred = x_pred/255
x_pred = x_pred.astype('float32')

#2. Modeling
model = tf.keras.Sequential([
                               tf.keras.applications.InceptionV3(weights=None, include_top=False, input_shape=(256, 256, 1)),
                               tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dense(526, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions'),
                               tf.keras.layers.Flatten(),
                               tf.keras.layers.Dense(64, activation='relu'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Dropout(0.2),
                               tf.keras.layers.Dense(26, activation='softmax')
                               ])
model.summary()

#3. Compile, Train

train_generator = idg.flow(x_train, y_train, batch_size=64)
test_generator = idg2.flow(x_test, y_test, batch_size=64)
valid_generator = idg2.flow(x_valid, y_valid)
pred_generator = idg2.flow(x_pred)

es = EarlyStopping(monitor='val_loss', patience=40, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.4, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(train_generator, epochs=10, validation_data=valid_generator, verbose=1, callbacks=[es, lr])

#4. Evaluate, Predict
loss, acc = model.evaluate_generator(test_generator)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict_generator(pred_generator)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
print(y_pred.shape) # (5000, 26)


sub.iloc[:,1:] = y_pred

sub.to_csv('C:/data/dacon_mnist2/anwsers/0209_base.csv', index=False)
print(sub.head())
