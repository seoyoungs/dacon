import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

######################################################

# 데이터 로드
train = pd.read_csv('../data/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

sub = pd.read_csv('../data/DACON_vision1/submission.csv')
print(sub.shape) # (20480, 2)

test = pd.read_csv('../data/DACON_vision1/test.csv')
print(test.shape)   # (20480, 786)

######################################################

#1. DATA
# print(train, test, sub)

# print(train['digit'].value_counts())    # 0부터 9까지

train2 = train.drop(['id', 'digit','letter'],1)
test2 = test.drop(['id','letter'],1)  # >> x_pred

train2 = train2.values  # >>> x
test2 = test2.values    # >>> x_pred

# plt.imshow(train2[100].reshape(28,28))
# plt.show()

train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# preprocess
train2 = train2/255.0
test2 = test2/255.0

#  ImageDataGenerator >> 데이터 증폭 : 데이터 양을 늘림으로써 오버피팅을 해결할 수 있다.
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
# width_shift_range : 왼쪽 오른쪽으로 움직인다.
# height_shift_range : 위쪽 아래쪽으로 움직인다.
idg2 = ImageDataGenerator()


# cross validation
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)

#2. Modeling
# %%time

reLR = ReduceLROnPlateau(patience=100, verbose=1, factor=0.5)
es = EarlyStopping(patience=120, verbose=1)

val_loss_min = []
val_acc_max = []
result = 0
nth = 0

for train_index, test_index in skf.split(train2, train['digit']) : # >>> x, y
    path = '../data/DACON_vision1/cp/0203_4_cp.hdf5'
    mc = ModelCheckpoint(path, save_best_only=True, verbose=1)

    x_train = train2[train_index]
    x_test = train2[test_index]
    y_train = train['digit'][train_index]
    y_test = train['digit'][test_index]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=47)

    train_generator = idg.flow(x_train, y_train, batch_size=16)
    test_generator = idg2.flow(x_test, y_test, batch_size=16)
    valid_generator = idg2.flow(x_valid, y_valid)
    pred_generator = idg2.flow(test2, shuffle=False)

    print(x_train.shape, x_test.shape, x_valid.shape)  # (1896, 28, 28, 1) (52, 28, 28, 1) (100, 28, 28, 1)
    print(y_train.shape, y_test.shape, y_valid.shape)  # (1896,) (52,) (100,)

    #2. Modeling
    model = Sequential()

    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28, 28,1), padding='same'))
    model.add(BatchNormalization()) 
    # BatchNormalization >> 학습하는 동안 모델이 추정한 입력 데이터 분포의 평균과 분산으로 normalization을 하고자 하는 것
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    #3. Compile, Train
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['acc'])
                                                                        # epsilon : 0으로 나눠지는 것을 피하기 위함
    learning_hist = model.fit_generator(train_generator, epochs=1000, validation_data=valid_generator, callbacks=[es, mc, reLR] )
    model.load_weights('../data/DACON_vision1/cp/0203_4_cp.hdf5')

    #4. Evaluate, Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)

    result += model.predict_generator(pred_generator, verbose=True)/40

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)
    val_loss_min.append(hist['val_loss'].min())
    val_acc_max.append(hist['val_acc'].max())

    nth += 1
    print(nth, "번째 학습을 완료했습니다.")

    print("val_loss_min :", np.mean(val_loss_min))  # val_loss_mean : 0.1835539501160383
    print("val_acc_max :", np.mean(val_acc_max))    # val_acc_max : 0.9512500002980232
    model.summary()

sub['digit'] = result.argmax(1)
print(sub)
sub.to_csv('../data/DACON_vision1/0203_4_private3.csv', index=False)