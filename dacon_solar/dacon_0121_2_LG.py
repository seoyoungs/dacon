###트레인 7일치씩 해서 다음 2일치를 예측하기
## cpu로 돌아간다.

import numpy as np
import pandas as pd
import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

train= pd.read_csv('C:/data/dacon_data/train/train.csv', encoding='cp949')
# print(train.head())
#337 7일차 337행
# print(train.shape) # (52560,8)

submission = pd.read_csv('C:/data/dacon_data/sample_submission.csv', encoding='cp949')
# print(submission.tail())


def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
df_train.iloc[:48]
# print(df_train.head())

train.iloc[48:96]
train.iloc[48+48:96+48]

# print(df_train.tail())

df_test = []

for i in range(81):
    file_path = 'C:/data/dacon_data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
# print(x_test.shape) #(3888, 7)

# print(x_test.head(48))
# print(df_train.head())
df_train.iloc[-48:]

from sklearn.model_selection import train_test_split
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(
    df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(
    df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

# print(x_train_1.head())
# print(x_test.head())

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


###########LGBM
from lightgbm import LGBMRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, x_train, y_train, x_valid, y_valid, x_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,                  
                         n_estimators=3300, bagging_fraction=0.7, learning_rate=0.1,
                         max_depth=4, subsample=0.7, feature_fraction=0.9, boosting_type='gbdt',
                         colsample_bytree=0.5, reg_lambda=5, n_jobs=-1)
                         
    model.fit(x_train, y_train, eval_metric = ['quantile'], 
          eval_set=[(x_valid, y_valid)], early_stopping_rounds=400, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(x_test).round(3)) #소수점 3번째 자리까지
    return pred, model


# Target 예측
def train_data(x_train, y_train, x_valid, y_valid, x_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, x_train, y_train, x_valid, y_valid, x_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = train_data(x_train_1, y_train_1, x_val_1, y_val_1,x_test)
results_1.sort_index()[:48]

# Target2
models_2, results_2 = train_data(x_train_2, y_train_2, x_val_2, y_val_2, x_test)
results_2.sort_index()[:48]

# print(results_1.shape, results_2.shape)
submission.iloc[:48]
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
# print(submission)
submission.iloc[:48]
submission.iloc[48:96]

submission.to_csv('C:/data/dacon_data/sub_0121_LGBM_1.csv', index=False)

'''
1.9991544728 --> 데이콘 점수 
'''
