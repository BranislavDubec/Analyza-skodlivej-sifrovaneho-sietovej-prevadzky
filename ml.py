__author__ = "Branislav Dubec"
__credits__ = ["Petr Chmelar"]
__version__ = "1.0.0"


import tensorflow as tf

import keras
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import seaborn as sns
import pydot

import pickle
import matplotlib.pyplot as plt
import time
NAME = "Blacklist-{}".format(int(time.time()))

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME),histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",)
NAME1 = "Blacklist-{}".format(int(time.time()))

tensorboard1 = tf.keras.callbacks.TensorBoard(log_dir='logs1/{}'.format(NAME),histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",)

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




def get_rm_model():
  model = tf.keras.Sequential([
    tf.keras.Input(shape=(80,), ),
      tf.keras.layers.Dense(2048, activation='relu'),
      tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
  ])

  model.compile(keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

def getSVM():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(80,),),
            tf.keras.layers.experimental.RandomFourierFeatures(
                output_dim=4096, scale=10.0, kernel_initializer="gaussian"
            ),
            tf.keras.layers.Dense(units=1024, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.sparse_categorical_crossentropy, #'hinge'
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)
    return model
#for root, dirs, files in os.walk('csv'):
 #   for name in files:
  #      filepath = root + os.sep + name
   #     normDataFrame( pd.read_csv(filepath, low_memory=False))
    #    print(filepath)
train = pd.read_csv("csv\\2015-03-19_capture-win.csv") # norm:470 bl:85
# for root, dirs, files in os.walk('csv'):
#     for name in files:
#         filepath = root + os.sep + name
#         if filepath.startswith("csv"):
#             train = pd.read_csv(filepath)
#             print(filepath)
#             normDataFrame(train)

test = pd.read_csv("csv\\2021-01-21-traffic-analysis-exercise.csv")
train = normDataFrame(train)
test = normDataFrame(test)
#normalizacia?




min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(train.values)
train = pd.DataFrame(x_scaled, columns=columns)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(test.values)
test = pd.DataFrame(x_scaled, columns=columns)
testB=test.pop('blacklisted')
test = test.to_numpy()
feature = train.columns
trainBlacklisted = train.pop('blacklisted')
print(trainBlacklisted.shape)
train1Black = trainBlacklisted

train= train.to_numpy()

train1 = train

model = getSVM()
print(model.summary())
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


model.fit(x=train, y=trainBlacklisted, validation_split=0.1,batch_size=1, epochs=20, verbose=2, callbacks=[tensorboard])
#model_graph = tf.keras.utils.model_to_dot(model, expand_nested=True, subgraph=True)
#graph = pydot.graph_from_dot_data(model_graph)
#print(model_graph.get_nodes())
#model_graph.write_raw('model.dot')
#model_graph.write_png('model.png')



model1 = get_rm_model()
print(model1.summary())
keras.utils.plot_model(model1, "my_2nd_model_with_shape_info.png", show_shapes=True)
print(train1.shape,train1Black.shape)
model1.fit(x=train1, y=train1Black, validation_split=0.1,batch_size=1, epochs=20, verbose=1, callbacks=[tensorboard])

testB = testB.to_numpy()

y_0 = model.predict_classes(test)
y_1 = model1.predict_classes(test)
model_ac = model1_ac = 0
false_pos = false_pos1 = 0
false_neg = false_neg1 = 0
for i in range(len(test)):

    try:
        print("X=%d, Predicted=%s Real=%s" % (i, y_0[i], testB[i]))

        if (y_0[i] == 1 and  int(testB[i] == 0)):
            false_neg = false_neg + 1
        elif (y_0[i] == 0 and  int(testB[i] == 1)):
            false_pos = false_pos + 1
        else:
            model_ac = model_ac + 1
        print("X=%d, Predicted=%s Real=%s"% (i, y_1[i], testB[i]))


        if (y_1[i] == 1 and  int(testB[i] == 0)):
            false_neg1 = false_neg1 + 1
        elif (y_1[i] == 0 and  int(testB[i] == 1)):
            false_pos1 = false_pos1 + 1
        else:
            model1_ac = model1_ac + 1
    except:
        pass

print("acc model = ", model_ac/len(test))
print("acc model = ", model1_ac/len(test))

print("false_pos model = ", false_neg/len(test))
print("false_pos model = ", false_neg1/len(test))

print("false_neg model = ", false_pos/len(test))
print("false_neg model = ", false_pos1/len(test))

#score=model.evaluate(norm_test, y_test, verbose=2)
#score1=model1.evaluate(norm_test, y_test, verbose=2)
#model.evaluate(dataset_t)

