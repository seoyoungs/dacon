import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

pred1 = pd.read_csv('C:/data/dacon_mnist2/anwsers/sub1.csv')
pred2 = pd.read_csv('C:/data/dacon_mnist2/anwsers/sub2.csv')
pred3 = pd.read_csv('C:/data/dacon_mnist2/anwsers/sub3.csv')


submission = pd.read_csv('C:/data/dacon_mnist2/sample_submission.csv')
submission.head()

pred1 = pred1.iloc[:,0:-1]
pred2 = pred2.iloc[:,0:-1]
pred3 = pred3.iloc[:,0:-1]
print(pred3)
sumsum = pd.concat([pred1, pred2, pred3], axis=1)

'''
submission['a', 'b', 'c', 'd', 'e', 
     'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 
     'x', 'y', 'z'] = pred1
submission['a', 'b', 'c', 'd', 'e', 
     'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 
     'x', 'y', 'z'] = pred2
submission['a', 'b', 'c', 'd', 'e', 
     'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 
     'x', 'y', 'z'] = pred3
'''
from collections import Counter
for i in range(sumsum.shape[0]) :
    predicts = sumsum.loc[i, : ]
    sub = Counter(predicts).most_common(n=25)[0][0]

'''
    submission.at[i, 'a', 'b', 'c', 'd', 'e', 
     'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 
     'x', 'y', 'z'] = Counter(predicts).most_common(n=25)#[0][0]
'''


# print(submission.head())

submission = submission[['index',  'a', 'b', 'c', 'd', 'e', 
     'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 
     'x', 'y', 'z']]
print(submission.head())

submission.to_csv('C:/data/dacon_mnist2/anwsers/merge1.csv', index=False)

