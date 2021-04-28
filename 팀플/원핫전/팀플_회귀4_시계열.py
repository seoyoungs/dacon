import numpy as np
import db_connect as db
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 

query = "SELECT * FROM main_data_table"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())

# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

# train_value = df[ '2020-09-01' >= df['date'] ]

# x_train = train_value.iloc[:,1:-1].astype('float32').to_numpy()
# y_train = train_value.iloc[:,-1].astype('float32').to_numpy()

# # x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)

# test_value = df[df['date'] >=  '2020-09-01']

# x_test = test_value.iloc[:,1:-1].astype('float32').to_numpy()
# y_test = test_value.iloc[:,-1].astype('float32').to_numpy()

# x_test=x_test.reshape(x_test.shape[0], x_test.shape[1],1)


from statsmodels.tsa.stattools import kpss

# df['date'] = pd.to_datetime(df['date']) #str to pandas Timestamp 

# df.index = df['date']
# df.set_index('date',inplace=True) #index로 변환

# 날짜에 따른 주문 건수 변화
uni_data = df['value']
uni_data.index = df['date']
uni_data.plot(subplots = True)
plt.show()
# print(df.tail())

# def kpss_test(series, **kw):    
#     statistic, p_value, n_lags, critical_values = kpss(df.values)
    
#     # Format Output
#     print(f'KPSS Statistic: {statistic}')
#     print(f'p-value: {p_value}')
#     print(f'num lags: {n_lags}')
#     print('Critial Values:')
    
#     for key, value in critical_values.items():
#         print(f'   {key} : {value}')
#     print(f'Result: The df is {"not " if p_value < 0.05 else ""} stationary')
    
# kpss_test(df['value'])

'''
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  
         train_size=0.8, random_state = 77, shuffle=True ) 
'''


