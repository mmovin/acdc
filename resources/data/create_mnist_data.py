from datetime import datetime
from pathlib import Path

import numpy as np
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 111


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flatt = x_train.reshape(x_train.shape[0], 784)
x_train_flatt = x_train_flatt.astype(float)
features_and_label = [str(x) for x in range(0,784)] + ['label']
print(x_train_flatt.shape, y_train.shape)
df = pandas.DataFrame(np.c_[x_train_flatt, y_train], columns=features_and_label)
#index = df['label'] == '4'
df = df[df['label'].isin([4, 9])].reset_index(drop=True)
df['label'].replace({4: 0, 9: 1}, inplace=True)


## Save the data:

df_train, df_test = train_test_split(df, random_state=RANDOM_SEED)

df_train.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)

## Save the data:
data_directory = str(datetime.date(datetime.now()))
Path(data_directory).mkdir(parents=True, exist_ok=True)

df_train.to_csv('{}/mnist_train.csv'.format(data_directory), index=False)
df_test.to_csv('{}/mnist_test.csv'.format(data_directory), index=False)