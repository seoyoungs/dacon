# https://www.kaggle.com/gulsahdemiryurek/image-classification-with-logistic-regression/data
# 위에는 노가다인데 정 안되면 하자....
# https://www.kaggle.com/itokianarafidinarivo/6000-store-items-images-classified-by-color/data?select=train
# 색깔 구분하기
# https://www.kaggle.com/phsaikiran/the-simpsons-characters-classification?select=submission.csv
# 심슨 얼굴 구별하기

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os
import time

        
# Load CNN packages 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)

train = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/data/LPD_competition/train",
    validation_split=VAL_SPLIT,
    subset="training",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="categorical"
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/data/LPD_competition/train",
    validation_split=VAL_SPLIT,
    subset="validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="categorical"
)

classes = train.class_names

# Take a look

# plt.figure(figsize=(10, 10))
# for images, labels in train.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(classes[np.argmax(labels[i])])
#         plt.axis("off")

print("Classes:", classes)

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

augmented_train = train.map(lambda x, y: (data_augmentation(x, training=True), y))

# Configure the dataset for performance
#train = train.prefetch(buffer_size=32)
augmented_train = augmented_train.prefetch(buffer_size=32)
validation = validation.prefetch(buffer_size=32)

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(classes), activation='softmax'))

print("Input shape:", IMG_SIZE + (3,))

model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

EPOCHS = 50
CALLBACKS = [keras.callbacks.ModelCheckpoint("../data/modelCheckpoint/lotte_0316_1_{epoch:02d}-{val_loss:.4f}.hdf5")]

# Training
history = model.fit(
    augmented_train,
    epochs=EPOCHS,
    callbacks=CALLBACKS,
    validation_data=validation,
)

directory = 'C:/data/LPD_competition/test'
columns = ['filename', 'prediction']
test = pd.DataFrame(None, columns=columns)

# Get in the directoy
for _, _, filenames in os.walk(directory):
    # Loop over the files
    for filename in filenames:
        # Get filepath
        filepath = os.path.join(directory, filename)
        
        # Get predictions
        raw_img = keras.preprocessing.image.load_img(filepath, target_size=IMG_SIZE)  # Load the image
        img_array = keras.preprocessing.image.img_to_array(raw_img)  # Convert to numpy array
        img_array = tf.expand_dims(img_array, 0)  # Reshaping - create batch axis
        predictions = model.predict(img_array)  # Make predictions
        prediction = classes[np.argmax(predictions)]  # Get the color with max probabilities
        
        # Add predictions on test data
        row = pd.DataFrame(data=[[filename, prediction]], columns=columns)
        test = test.append(row)

test.to_csv('C:/data/csv/lotte0317_1.csv', index=False)
