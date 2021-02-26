import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
# https://keras.io/api/preprocessing/image/

csv_train = pd.read_csv('C:/data/dacon_mnist/train.csv')
csv_test = pd.read_csv('C:/data/dacon_mnist/test.csv')

import tensorflow as tf

model_1 = tf.keras.models.load_model('C:/data/modelCheckpoint/0203_1_model_1.h5', compile=False)
model_2 = tf.keras.models.load_model('C:/data/modelCheckpoint/0203_1_model_2.h5', compile=False)
model_3 = tf.keras.models.load_model('C:/data/modelCheckpoint/0203_1_model_3.h5', compile=False)

os.mkdir('C:/data/dacon_mnist/images_test/none')
os.mv('C:/data/dacon_mnist/images_test/*.png images_test/none')


datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory('./images_test', target_size=(224,224), color_mode='grayscale', class_mode='categorical', shuffle=False)

predict_1 = model_1.predict_generator(test_generator).argmax(axis=1)
predict_2 = model_2.predict_generator(test_generator).argmax(axis=1)
predict_3 = model_3.predict_generator(test_generator).argmax(axis=1)

submission = pd.read_csv('./data/submission.csv')
submission.head()

submission["predict_1"] = predict_1
submission["predict_2"] = predict_2
submission["predict_3"] = predict_3
# submission.head()

from collections import Counter

for i in range(len(submission)) :
    predicts = submission.loc[i, ['predict_1','predict_2','predict_3']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]

submission.head()

submission = submission[['id', 'digit']]
submission.head()

submission.to_csv('submission_ensemble_3.csv', index=False)

# 제출용 파일
submission = pd.read_csv('./data/submission.csv')
submission.head()

submission["predict_1"] = predict_1
submission["predict_2"] = predict_2
submission["predict_3"] = predict_3
submission.head()

from collections import Counter

for i in range(len(submission)) :
    predicts = submission.loc[i, ['predict_1','predict_2','predict_3']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]

submission.head()

submission = submission[['id', 'digit']]
submission.head()

submission.to_csv('C:/data/dacon_mnist/answer/0204_1_mnist.csv', index=False)




