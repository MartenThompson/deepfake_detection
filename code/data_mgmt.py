
import numpy as np
import os
import pandas as pd
import tensorflow as tf

BATCH_SIZE = 10
IMG_HW = 160

DIR = '/Users/MartenThompson/Git/deepfake_detection/data_ignore/'
folders = os.listdir(DIR)

if '.DS_Store' in folders:
    folders.remove('.DS_Store')


folder_sizes = []
for name in folders:
    n = len(os.listdir(DIR+'/'+name))
    
    if '.DS_Store' in os.listdir(DIR+'/'+name):
        n -= 1
    
    folder_sizes.append(n)

print(sum(folder_sizes))

metadata = pd.read_csv('/Users/MartenThompson/Git/deepfake_detection/metadata.csv')
metadata['label'].replace({'FAKE': 1, 'REAL': 0}, inplace=True)
labels = metadata['label'].values
labels = np.repeat(labels, folder_sizes).tolist()

print(len(labels))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/Users/MartenThompson/Git/deepfake_detection/data_ignore',
    #labels=labels,
    validation_split = 0.2,
    subset='training',
    seed=444,
    image_size=(IMG_HW, IMG_HW),
    batch_size=BATCH_SIZE)


