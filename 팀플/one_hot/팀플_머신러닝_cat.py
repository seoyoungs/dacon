# [CatBoost]
import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from time import time
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor


# db 직접 불러오기 

# 0 없다
# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"

# 0 있다
query = "SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC  "
query1 = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())
db.cur.execute(query1)
dataset1 = np.array(db.cur.fetchall())

# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)
df1 = pd.DataFrame(dataset1, columns=column_name)

db.connect.commit()

train_value = df[ '2020-09-01' > df['date'] ]

x_train1 = train_value.iloc[:,1:-1].astype('int64')
y_train1 = train_value['value'].astype('int64').to_numpy()

test_value = df1[df1['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64')
y_pred = test_value['value'].astype('int64').to_numpy()

x_train1 = pd.get_dummies(x_train1, columns=["category", "dong"]).to_numpy()
x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()

# print(x_train1.shape, y_train1.shape) #(3472128, 6) (3472128,)
# print(x_pred.shape, y_pred.shape)   #(177408, 6) (177408,)


kfold = KFold(n_splits=5, shuffle=True)


r2_list = []
rmse_list = []
mae_list = []
time_list = []

num = 0

# 훈련 loop
for train_index, valid_index in kfold.split(x_train1):

    # print(train_index, len(train_index))    #2777702
    # print(valid_index, len(valid_index))    #694426

    x_train = x_train1[train_index]
    x_valid = x_train1[valid_index]
    y_train = y_train1[train_index]
    y_valid = y_train1[valid_index]
 
    #2. 모델구성
    model = CatBoostRegressor(iterations=300,
                        learning_rate=0.1,
                        loss_function='RMSE',
                        depth=4,
                        task_type="GPU",
                        devices='0:1')

    start_time = timeit.default_timer()

    #3. 훈련
    model.fit(x_train, y_train, verbose = True, eval_set=(x_valid, y_valid))

    finish_time = timeit.default_timer()
    time = round(finish_time - start_time, 2)
    time_list.append(time)
    print(f'{num}fold time : ', time)
 

    # 모델저장
    joblib.dump(model, f'../data/modelCheckpoint/cat_kfold_{num}.pkl')

    # 모델로드
    model = joblib.load(f'../data/modelCheckpoint/cat_kfold_{num}.pkl')

    #4. 평가, 예측
    y_predict = model.predict(x_pred)
    print('예측값 : ', y_predict[:5])
    print('실제값 : ', y_pred[:5])


    # r2_list
    r2 = r2_score(y_pred, y_predict)
    print('r2 score     :', r2)
    r2_list.append(r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    print('rmse score     :', rmse)
    rmse_list.append(rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    print('mae score     :', mae)
    mae_list.append(mae) 

    num += 1


# 엑셀 추가 코드 
# 경로 변경 필요!!!!
df = pd.DataFrame(y_predict)
df['test'] = y_pred
df.to_csv('../data/csv/deltime_data_cat.csv',index=False)

print('======================================')
r2_list = np.array(r2_list)
print('r2_list : ', r2_list)
rmse_list = np.array(rmse_list)
print('rmse_list : ',rmse_list)
mae_list = np.array(mae_list)
print('mae_list : ',mae_list)
time_list = np.array(time_list)
print('time_list : ', time_list)





