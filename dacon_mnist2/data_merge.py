
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

pred1 = pd.read_csv('C:/data/dacon_mnist2/answers/2015.csv')
pred2 = pd.read_csv('C:/data/dacon_mnist2/answers/2016.csv')
pred3 = pd.read_csv('C:/data/dacon_mnist2/answers/2019.csv')
pred4 = pd.read_csv('C:/data/dacon_mnist2/answers/2020.csv')

submission = pd.read_csv('C:/data/dacon_mnist/submission.csv')
submission.head()

submission["pred_1"] = pred1
submission["pred_2"] = pred2
submission["pred_3"] = pred3
submission["pred_4"] = pred4


from collections import Counter
for i in range(len(submission)) :
    predicts = submission.loc[i, ['pred_1','pred_2','pred_3','pred_4','pred_5']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]


print(submission.head())

submission = submission[['id', 'digit']]
print(submission.head())

submission.to_csv('C:/data/dacon_mnist/answer/merge2.csv', index=False)


