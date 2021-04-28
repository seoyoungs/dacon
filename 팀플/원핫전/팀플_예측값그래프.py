    import matplotlib.pyplot as plt
 
    fig = plt.figure(figsize=(10,10))
    fig.set_facecolor('white')
    ax = fig.add_subplot()
    ax.plot(y_pred)
    ax.plot(y_predict)
    plt.xticks(rotation=45)
    plt.show()











import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from tensorflow.keras.layers import LeakyReLU, PReLU
from time import time







acti_list = ['elu', 'relu', 'selu','tanh','swish']
opti_list = [RMSprop, Nadam, Adam, Adadelta, SGD]
acti = acti_list[2]
opti = opti_list[1]
batch = 64
lrr = 0.001
epo = 50






