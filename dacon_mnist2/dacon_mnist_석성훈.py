import numpy as np
import pandas as pd
import PIL
from numpy import asarray
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
import scipy.signal as signal

# np.save('c:/data/test/dirty_mnist/temporary/test2.npy', arr=img_ch_np)
# np.save('c:/data/test/dirty_mnist/temporary/test4.npy', arr=img_ch_np2)
x_data = np.load('c:/data/test/dirty_mnist/temporary/test2.npy')
x_test = np.load('c:/data/test/dirty_mnist/temporary/test4.npy')

x_data = x_data.reshape(50000, 128, 128, 1)
x_test = x_test.reshape(5000, 128, 128, 1)

dataset = pd.read_csv('c:/data/test/dirty_mnist/dirty_mnist_2nd_answer.csv')
submission = pd.read_csv('c:/data/test/dirty_mnist/sample_submission.csv')
import matplotlib.pyplot as plt
y_data = dataset.iloc[:,:]
print(y_data)

# plt.figure(figsize=(20, 5))
# ax = plt.subplot(2, 10, 1)
# plt.imshow(x_test[0])

# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.show()

from tensorflow.keras.optimizers import Adam
def convmodel():
    model = Sequential()
    model.add(Conv2D(128, 8, padding='same', activation='relu', input_shape=(128,128,1)))
    model.add(Conv2D(64,8,padding='same',activation='relu'))
    model.add(AveragePooling2D(3))
    model.add(Conv2D(32,8,padding='same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    optimizer = Adam(lr=0.002)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for i in alphabet:
    y = y_data.loc[:,i]
    print(y)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y, train_size=0.8, shuffle=True, random_state=42)
    model = convmodel()
    checkpoint = ModelCheckpoint(f'c:/data/test/dirty_mnist/checkpoint/checkpoint-{i}.hdf5', 
    monitor='val_loss', save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(patience=3,verbose=1,factor=0.5) #learning rate scheduler
    es = EarlyStopping(patience=6, verbose=1)
    model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), batch_size=32, callbacks=[checkpoint,lr,es])
    model2 = load_model(f'c:/data/test/dirty_mnist/checkpoint/checkpoint-{i}.hdf5', compile=False)
    y_pred = model.predict(x_test)
    print(y_pred)
    y_recovery = np.where(y_pred<0.5, 0, 1)
    print(y_recovery)
    submission[i] = y_recovery
submission.to_csv('c:/data/test/dirty_mnist/submission.csv', index=False)

