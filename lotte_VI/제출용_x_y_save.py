import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

#########데이터 로드
cate_load =  'C:/data/LPD_competition/train/'

cate_merge = []
for i in range(0,1000) :
    i = "%d"%i
    cate_merge.append(i)
# 1000클래스 * 48개

len_classes = len(cate_merge)

image_w = 128
image_h = 128

pixels = image_h * image_w * 3

X = []
y = []
for idx, cat in enumerate(cate_merge):
    
    # 원 핫 반복 문
    label = [0 for i in range(len_classes)]
    label[idx] = 1

    image_dir = cate_load + "/" + cat
    file_path = glob.glob(image_dir+"/*.jpg")
    print(cat, " file length : ", len(file_path))
    for i, f in enumerate(file_path):
        img = Image.open(f)
        img = img.convert("RGB") # 컬러형태
        img = img.resize((image_w, image_h)) #사이즈 정비
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)
X = np.array(X)
y = np.array(y)

np.save("C:/data/npy/lotte_x_128.npy", arr=X)
np.save("C:/data/npy/lotte_y_128.npy", arr=y)


x = np.load("C:/data/npy/lotte_x_128.npy",allow_pickle=True)
y = np.load("C:/data/npy/lotte_y_128.npy",allow_pickle=True)
print('저장 완료')

print(x.shape)
print(y.shape)



# lotte_x_224
# lotte_x_255
