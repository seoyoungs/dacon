'''
LGBG
learning_rate을 줄이면서 n_estimator를 크게하는 것은 
부스팅에서의 기본적인 튜닝 방안

lightGBM에서 제공하는 파라미터
binary(Cross Entropy)
multiclass(Cross Entropy)
regression_l2(MSE)
regression_l1 (MAE)
mape (MAPE)
poisson (Log Transformation)
quantile (Quantile)
huber (Huber loss, MAE approx)
fair (Fair loss, MAE approx)
gamma (Residual Deviance)
lambdarank
tweedie


learning_rate : 일반적으로 0.01~ 0.1 정도로 맞추고 다른 파라미터를 튜닝한다
              : 나중에 성능을 더 높일 때 learning_rate를 더 줄인다.
num_iteration : 기본값이 100인데 1000정도는 해주는게 좋다. 너무 크게하면 
              : 과적합 발생할 수 있다.
max_depth : -1로 설정하면 제한없이 분기한다. feature가 많다면 크게 설정한다.
          : 파라미터 설정시 우선적으로 설정
boosting : 기본값은 gbdt이며 정확도가 중요할 때는 딥러닝의 드랍아웃과 같은
         : dart를 사용한다.
bagging_fraction: 배깅을 하기위해서 데이터를 랜덤 샘플링하여 학습에 사용한다. 
                  비율은 0 < fraction <= 1 이며 0이 되지 않게 해야한다.
feature_fraction: 1보다 작으면 LGBM은 매 iteration(tree)마다 다른 feature를
                랜덤하게 추출하여 학습하게 된다. 만약 0.8로 설정하면 매 tree를 구성할 때,
                feature의 80%만 랜덤하게 선택한다. 과적합을 방지하기 위해 사용할 수 있으며
                학습 속도가 향상된다.
scale_pos_weight : 클래스 불균형의 데이터 셋에서 weight를 주는 방식으로 positive를 증가시킨다.
                  기본값은 1이며 불균형의 정도에 따라 조절한다
early_stopping_round : Validation 셋에서 평가지표가 더 이상 향상되지 않으면 학습을 정지한다
                     평가지표의 햘상이 n round이상 지속되면 학습을 정지한다.        
lambda_l1, lambda_l2: 정규화를 통해 과적합을 방지할 수 있지만, 
                     정확도를 저하시킬수 있기 때문에 일반적으로 defalut 값인 0으로 둔다.         
'''

'''
import lightgbm as lgb
from lightgbm import LGBMRegressor
#LGBG
#learning_rate을 줄이면서 n_estimator를 크게하는 것은 
#부스팅에서의 기본적인 튜닝 방안
# Get the model and the predictions in (a) - (b)
def LGBM(q, x_train, y_train, x_valid, y_valid, x_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,n_estimators=10000, 
                          bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
                         
    model.fit(x_train, y_train, eval_metric = ['quantile'], 
          eval_set=[(x_valid, y_valid)], early_stopping_rounds=300, verbose=500)
    # (b) Predictions
    pred = pd.Series(model.predict(x_test).round(2))
    return pred, model
'''