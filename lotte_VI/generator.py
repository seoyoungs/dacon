import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import pandas as pd
# randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

#0. 변수
batch = 32
seed = 42
dropout = 0.2
epochs = 100
model_path = '../data/modelCheckpoint/lotte_0316_1_{epoch:02d}-{val_loss:.4f}.hdf5'
sub = pd.read_csv('C:/data/LPD_competition/sample.csv', header = 0)
es = EarlyStopping(patience = 5)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)


#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    'C:/data/LPD_competition/train',
    target_size = (256, 256),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    'C:/data/LPD_competition/train',
    target_size = (256, 256),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    'C:/data/LPD_competition/test',
    target_size = (256, 256),
    class_mode = None,
    batch_size = batch,
    seed = seed,
    shuffle = False
)


# ======================== model ===========================
from keras.optimizers import Adam
model = Sequential([
    # 1st conv
  Conv2D(96, (3,3),strides=(4,4), padding='same', activation='relu', input_shape=(64, 64, 3)),
  BatchNormalization(),
  MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  Conv2D(256, (3,3),strides=(1,1), activation='relu',padding="same"),
  BatchNormalization(),
     # 3rd conv
#   Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
#   BatchNormalization(),
#     # 4th conv
#   Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
#   BatchNormalization(),
    # 5th Conv
  Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  BatchNormalization(),
  MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  Flatten(),
#   # To FC layer 1
#   Dense(4096, activation='relu'),
  Dropout(0.5),
  #To FC layer 2
  Dense(15, activation='relu'),
  Dropout(0.5),
  Dense(1000, activation='softmax')
  ])

model.compile(
    optimizer=Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['acc']
   )
# hist = model.fit_generator(generator=xy_train,
#                     validation_data=test_set,
#                     steps_per_epoch=16,
#                     validation_steps=16,
#                     epochs=30)

model.fit_generator(
        train_data,
        steps_per_epoch=5,
        epochs=50,
        validation_data=test_data,
        validation_steps=5)

modelpath = '../data/modelCheckpoint/lotte_0316_1_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, mode='auto')
history = model.fit_generator(generator=train_data,
                    validation_data=val_data,
                    epochs=2,
                    callbacks=[es, cp, lr])
acc = history.history['acc']
val_acc = history.history['val_acc']
print("val_acc :", np.mean(val_acc))
print('acc: ', acc[-1])

# test_data.reset()
y_pred = model.predict_generator(test_data, steps=5) #a += b는 a= a+b
# 반복 수에 따라서 40으로 나누기
# predict_generator 예측 결과는 클래스별 확률 벡터로 출력
print('예측결과:', y_pred)

#제출========================================
sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = y_pred.argmax(1) # y값 index 2번째에 저장
sub
sub.to_csv('C:/data/csv/lotte0316_1.csv',index=False)






