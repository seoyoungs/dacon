###https://dacon.io/competitions/official/235680/codeshare/2289?page=1&dtype=recent&ptype=pub
########오늘 공부할 것 (learning late, keras 55_split)
##7,8일 타겟만 구하는거

import pandas as pd
import numpy as np

train = pd.read_csv('C:/data/태양광 발전량 예측data/train/train.csv', encoding='cp949', header=0, index_col=0)
submission = pd.read_csv('C:/data/태양광 발전량 예측data/sample_submission.csv')

submission.set_index('id',inplace=True)

#2. 전처리
def transform(dataset, target, start_index, end_index, history_size,
                      target_size, step):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index, 48):
        indices = range(i - history_size, i, step)
        data.append(np.ravel(dataset[indices].T))
        labels.append(target[i:i+target_size])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


x_col =['TARGET'] #왜 이렇게 잡았냐? 7,8일은 타겟값만 구하는게 목표
y_col = ['TARGET']

dataset = train.loc[:,x_col].values
label = np.ravel(train.loc[:,y_col].values)

past_history = 48 * 2   #30분씩 1일치는 48번, 48번을 2일치 예상(48*2)
future_target = 48 * 2

train_data, train_label = transform(dataset, label, 0,None, past_history,future_target, 1)

### transform test
test = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'C:/data/태양광 발전량 예측data/test/{i}.csv')
    tmp = tmp.loc[:, x_col].values
    tmp = tmp[-past_history:,:]
    data.append(np.ravel(tmp.T))
    data = np.array(data)
    test.append(data)
test = np.concatenate(test, axis=0)


# 3. 모델 학습 및 예측
from sklearn.ensemble import RandomForestRegressor
N_ESTIMATORS = 1000 #1000정도 해주는 것이 좋다.
rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    max_features=1, random_state=0,
                                    max_depth = 5,
                                    verbose=True,
                                    n_jobs=-1)
rf.fit(train_data, train_label)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    submission.iloc[:, i] = np.ravel(y_pred)

submission.to_csv(f'C:\data\csv\submission0119_3.csv', index=True)


