##### 가장 많이 나온 것 합치는 함수 만들기
## 우선 하기 전에 csv파일에서 id값 삭제 후 digit값만 추출하기
'''
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

pred1 = pd.read_csv('잘나온 파일경로 1')
pred2 = pd.read_csv('잘나온 파일경로 2')
pred3 = pd.read_csv('잘나온 파일경로 3')


predict_1 = pred1
predict_2 = pred2
predict_3 = pred3

submission = pd.read_csv('C:/data/dacon_mnist/submission.csv')
submission.head()

submission["predict_1"] = predict_1
submission["predict_2"] = predict_2
submission["predict_3"] = predict_3

from collections import Counter
for i in range(len(submission)) :
    predicts = submission.loc[i, ['predict_1','predict_2','predict_3']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]


print(submission.head())

submission = submission[['id', 'digit']]
print(submission.head())

submission.to_csv('저장할 파일 경로', index=False)
'''

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

pred1 = pd.read_csv('C:/data/dacon_mnist/answer/sub1.csv')
pred2 = pd.read_csv('C:/data/dacon_mnist/answer/sub2.csv')
pred3 = pd.read_csv('C:/data/dacon_mnist/answer/sub3.csv')
pred4 = pd.read_csv('C:/data/dacon_mnist/answer/sub4.csv')
pred5 = pd.read_csv('C:/data/dacon_mnist/answer/sub5.csv')

submission = pd.read_csv('C:/data/dacon_mnist/submission.csv')
submission.head()

submission["pred_1"] = pred1
submission["pred_2"] = pred2
submission["pred_3"] = pred3
submission["pred_4"] = pred4
submission["pred_5"] = pred5


from collections import Counter
for i in range(len(submission)) :
    predicts = submission.loc[i, ['pred_1','pred_2','pred_3','pred_4','pred_5']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]


print(submission.head())

submission = submission[['id', 'digit']]
print(submission.head())

submission.to_csv('C:/data/dacon_mnist/answer/merge2.csv', index=False)


