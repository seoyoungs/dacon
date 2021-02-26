## 꼭 뿌신다 내가

import pandas as pd
import os

csv_train = pd.read_csv('C:/data/dacon_mnist/train.csv')
csv_test = pd.read_csv('C:/data/dacon_mnist/test.csv')

csv_train.head()

# train 이미지들과 test 이미지들을 저장해놓을 폴더를 생성합니다.
os.mkdir('C:/data/dacon_mnist/image_train/0')
os.mkdir('C:/data/dacon_mnist/image_train/1')
os.mkdir('C:/data/dacon_mnist/image_train/2')
os.mkdir('C:/data/dacon_mnist/image_train/3')
os.mkdir('C:/data/dacon_mnist/image_train/4')
os.mkdir('C:/data/dacon_mnist/image_train/5')
os.mkdir('C:/data/dacon_mnist/image_train/6')
os.mkdir('C:/data/dacon_mnist/image_train/7')
os.mkdir('C:/data/dacon_mnist/image_train/8')
os.mkdir('C:/data/dacon_mnist/image_train/9')
os.mkdir('C:/data/dacon_mnist/image_train/9')
os.mkdir('C:/data/dacon_mnist/image_test')

import cv2

for idx in range(len(csv_train)) :
    img = csv_train.loc[idx, '0':].values.reshape(28, 28).astype(int)
    digit = csv_train.loc[idx, 'digit']
    cv2.imwrite(f'C:/data/dacon_mnist/image_train/{digit}/{csv_train["id"][idx]}.png', img)

for idx in range(len(csv_test)) :
    img = csv_test.loc[idx, '0':].values.reshape(28, 28).astype(int)
    cv2.imwrite(f'C:/data/dacon_mnist/image_test/{csv_test["id"][idx]}.png', img)

import tensorflow as tf

model_1 = tf.keras.applications.InceptionResNetV2(weights=None, include_top=True, input_shape=(224, 224, 1), classes=10)

model_2 = tf.keras.Sequential([
                               tf.keras.applications.InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 1)),
                               tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dense(1024, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions')
                               ])

model_3 = tf.keras.Sequential([
                               tf.keras.applications.Xception(weights=None, include_top=False, input_shape=(224, 224, 1)),
                               tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dense(1024, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions')
                               ])

model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                             rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

train_generator = datagen.flow_from_directory(
          'C:/data/dacon_mnist/image_train', target_size=(224,224), color_mode='grayscale', 
             class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(
        'C:/data/dacon_mnist/image_train', target_size=(224,224), color_mode='grayscale', 
           class_mode='categorical', subset='validation')


checkpoint_1 = tf.keras.callbacks.ModelCheckpoint(f'C:/data/modelCheckpoint/0203_1_model_1.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(f'C:/data/modelCheckpoint/0203_1_model_2.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(f'C:/data/modelCheckpoint/0203_1_model_3.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)

model_1.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint_1])
model_2.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint_2])
model_3.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint_3])
# 출력 결과를 보여드리기 위해서 임시로 epochs=20 으로 설정하고 실행하였습니다.
# 실제로는 epochs=500 으로 설정하고 실행해야 하며, 많은 시간이 소요됩니다.

