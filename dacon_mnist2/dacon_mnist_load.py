import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

image_set=np.load('C:/data/dacon_mnist2/npy_data/a.npy')

sub=pd.read_csv('C:/data/dacon_mnist2/sample_submission.csv')
answer=pd.read_csv('C:/data/dacon_mnist2/dirty_mnist_answer.csv')

answer=answer.iloc[:25000, :]
answer=answer.to_numpy()

# print(image_set.shape) # (50000, 256, 256)
# print(answer.info()) # (50000, 27)

image_set=image_set.reshape(-1, 256, 256, 1)/255
kf=KFold(n_splits=5, shuffle=True, random_state=23)

for train_index, test_index in kf.split(image_set, answer):
    x_train=image_set[train_index]
    x_test=image_set[test_index]
    y_train=answer[train_index]
    y_test=answer[test_index]

# for train_index, val_index in kf.split(x_train, y_train):
#     x_train=x_train[train_index]
#     x_val=x_train[val_index]
#     y_train=y_train[train_index]
#     y_val=y_train[val_index]

print(x_train.shape)
# print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
# print(y_val.shape)
print(y_test.shape)