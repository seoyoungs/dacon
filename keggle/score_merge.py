import numpy as np
import pandas as pd
import timeit
import tensorflow as tf


x = []
for i in range(6):
    df = pd.read_csv(f'C:/data/keggle/taita/answer/submission{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'C:/data/keggle/taita/answer/submission{i}.csv', index_col=0, header=0)
for i in range(10000):
    for j in range(1):
        a = []
        for k in range(6):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype(int)
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('C:/data/keggle/taita/answer/submission2_fff_0429.csv')




