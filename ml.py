# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv (r'csv\2015-03-19_capture-win.csv')
test_d = pd.read_csv(r'csv\2017-05-02_normal.csv')
columns = ['duration','srcPort','dstPort','service','srcBytes', 'dstBytes',
           'flag', 'land', 'urgent', 'ja3Ver']
columns.extend(['ja3Cipher' + str(i) for i in range(36)])
columns.extend(['ja3Extension' + str(i) for i in range(26)])
columns.extend(['ja3Ec' + str(i) for i in range(6)])
columns.extend(['ja3Ecpf' + str(i) for i in range(2)])
columns.extend(['blacklisted'])
def normDataFrame(df):
    df.drop(df[df['ja3']  == "0"].index , inplace=True)
    df = df.drop('srcIp' , axis=1)
    df = df.drop('dstIp', axis=1)
    data = []
    ct = 0
    ct_b = 0
    ct_g = 0
    new_df = pd.DataFrame(columns = columns, dtype=np.float64)
    for index, row in df.iterrows():


        data.append(float(row['duration']))
        data.append(float(row['srcPort']))
        data.append(float(row['dstPort']))
        data.append(float(row['service']))
        data.append(float(row['srcBytes']))
        data.append(float(row['dstBytes']))
        data.append(float(row['flag']))
        data.append(float(row['land']))
        data.append(float(row['urgent']))
        data.append(float(row['ja3Ver']))

        ciphers = row['ja3Cipher'].split('-')
        for cipher in ciphers:
            data.append(float(cipher))
        extensions = row['ja3Extension'].split('-')
        for extension in extensions:
            data.append(float(extension))
        ecs = row['ja3Ec'].split('-')
        for ec in ecs:
            data.append(float(ec))
        ecpfs = row['ja3Ecpf'].split('-')
        for ecpf in ecpfs:
            data.append(float(ecpf))
        data.append(float(row['blacklisted']))
        if(row['blacklisted'] == 0):
            ct_b = ct_b+1
        else:
            ct_g = ct_g+1
        new_df.loc[len(new_df)] = data
        data = []
        ct = ct + 1
    print(ct_b,ct_g)
    return new_df




def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

#test = normDataFrame(df)

train = normDataFrame(test_d)
exit()
#normalizacia?
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(test.values)
#test = pd.DataFrame(x_scaled, columns=columns)
print(test)
test = tf.keras.utils.normalize(test)
print(test)
exit()
blacklisted = test.pop('blacklisted')
dataset = tf.data.Dataset.from_tensor_slices((test.values, blacklisted.values))

blacklisted_test = train.pop('blacklisted')
dataset_t = tf.data.Dataset.from_tensor_slices((train.values, blacklisted_test.values))
train_dataset = dataset.shuffle(len(test)).batch(32)
model = get_compiled_model()
model.fit(train_dataset, epochs=15)
#model.evaluate(dataset_t)

