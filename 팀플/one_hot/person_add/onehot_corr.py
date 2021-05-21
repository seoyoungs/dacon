import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
import tensorflow.keras.backend as K
import scipy.stats as stats

# 0 있다
query = "SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC  "
# query1 = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())
# db.cur.execute(query1)
# dataset1 = np.array(db.cur.fetchall())

# pandas 넣기
column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity','person', 'value']

df = pd.DataFrame(dataset, columns=column_name)
# df1 = pd.DataFrame(dataset1, columns=column_name)

db.connect.commit()

# df = pd.DataFrame(dataset, columns=column_name)

train_value = df[ '2020-09-01' > df['date'] ]

# x_train1 = train_value.iloc[:,1:-1].astype('int64')
x_train1 = train_value.iloc[:,1:].astype('float64')
y_train1 = train_value['value'].astype('int64').to_numpy()

# test_value = df1[df1['date'] >=  '2020-09-01']

# x_pred = test_value.iloc[:,1:-1].astype('int64')
# y_pred = test_value['value'].astype('int64').to_numpy()

# x_train1 = pd.get_dummies(x_train1, columns=["category", "dong"]).to_numpy()
# x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()

# 스피어만 상관계수 검정
corr = x_train1.corr(method='spearman')
print(corr)




