import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import string
import scipy.signal as signal
from keras.applications.resnet import ResNet101,preprocess_input


img1=[]
for i in range(0,72000):
    file='C:/data/LPD_competition/test/%d.jpg'%i
    img2=Image.open(file)
    img2 = img2.convert('RGB')
    img2 = img2.resize((128,128))
    img_data2=asarray(img2)
    img1.append(img_data2)    

np.save('C:/data/npy/lotte_pred_128.npy', arr=img1)
# alphabets = string.ascii_lowercase
# alphabets = list(alphabets)

# x = np.load('../data/csv/Dacon3/train4.npy')
x_pred = np.load('C:/data/npy/lotte_pred_128.npy',allow_pickle=True)

print(x_pred.shape)

# lotte_pred_224
# lotte_pred_255

