import pandas as pd

# csv 형식으로 된 데이터 파일을 읽어옵니다.
train = pd.read_csv('/content/drive/MyDrive/dacon/소득예측경진대회/train.csv')
train

print(train.shape)

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)

# 범주형인지 수치형인지 unique() 메소드 확인
print(train['workclass'].unique())
print(train['occupation'].unique())
print(train['native.country'].unique())

# 결측치를 처리하는 함수를 작성합니다.
def handle_na(data, missing_col):
    temp = data.copy()
    for col, dtype in missing_col:
        if dtype == 'O':
            # 범주형 feature가 결측치인 경우 해당 행들을 삭제해 주었습니다.
            temp = temp.dropna(subset=[col])  # 삭제  방법
    return temp

train = handle_na(train, missing_col)

# 결측치 처리가 잘 되었는지 확인해 줍니다.
missing_col = check_missing_col(train) 

# 데이터 전처리
#라벨인코딩을 하기 위함 dictionary map 생성 함수
def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            label_map = {'unknown':0}
            for i, key in enumerate(dataframe[col].unique()):
                label_map[key] = i  #새로 등장하는 유니크 값들에 대해 1부터 1씩 증가시켜 키값을 부여해줍니다.
            label_maps[col] = label_map
    return label_maps

# 각 범주형 변수에 인코딩 값을 부여하는 함수
def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            dataframe[col] = dataframe[col].map(label_map[col])
            #dataframe[col] = dataframe[col].fillna(label_map[col]['unknown']) #혹시 모를 결측값은 unknown의 값(0)으로 채워줍니다.
    return dataframe

train = label_encoder(train, make_label_map(train))

print(train)

# 변수 및 모델 정의(앙상블)
# 분석에 필요 없는 id 와 예측하고자 하는 값 타겟을 제거해줍니다.
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


X = train.drop(['id', 'target'], axis=1)
y = train['target']


#  모델 성능 확인
# 모델 선언 (k_fold로 반복 훈련)
# for fold in range(10):
# model = MLPClassifier(max_iter=12, hidden_layer_sizes=100, verbose=False)
# model  = XGBClassifier(solver='liblinear')
model = DecisionTreeClassifier(random_state = 42)

# 모델 학습
model.fit(X, y)

# 먼저 점수를 메기는 방법인 평가 지표(Metric)를 정의합니다.
import numpy as np

def ACCURACY(true, pred):   
    score = np.mean(true==pred)
    return score

# 모델의 예측과 실제 정답값을 비교합니다.
prediction = model.predict(X)

score = ACCURACY(y, prediction)

print(f"모델의 정확도는 {score*100:.2f}% 입니다")

# csv형식으로 된 데이터 파일을 읽어옵니다.
test = pd.read_csv('/content/drive/MyDrive/dacon/소득예측경진대회/test.csv')
test.head()

test = label_encoder(test, make_label_map(test))
test = test.drop(['id'],axis=1)
test.head()

# 전처리가 완료된 테스트 데이터셋을 통해 본격적으로 학습한 모델로 추론을 시작합니다.
prediction = model.predict(test)
prediction

# 제출용 Sample 파일을 불러옵니다
submission = pd.read_csv('/content/drive/MyDrive/dacon/소득예측경진대회/sample_submission.csv')
submission.head()

submission['target'] = prediction

# 데이터가 잘 들어갔는지 확인합니다
print(submission)

submission.to_csv('/content/drive/MyDrive/dacon/소득예측경진대회/predict_submit_7.csv', index=False)



