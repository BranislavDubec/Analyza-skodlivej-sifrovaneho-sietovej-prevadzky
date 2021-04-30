__author__ = "Branislav Dubec"
__credits__ = ["Petr Chmelar"]
__version__ = "1.0.0"


#import tensorflow as tf

#import keras
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import sklearn
from libsvm.svmutil import *
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import scikitplot as skplt
import time

NAME = "Blacklist-{}".format(int(time.time()))

# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME),histogram_freq=0,  # How often to log histogram visualizations
#     embeddings_freq=0,  # How often to log embedding visualizations
#     update_freq="epoch",)


columns = ['duration','srcPort','dstPort','service','srcBytes', 'dstBytes',
           'flag', 'land', 'urgent', 'ja3Ver']

columns.extend(['ja3Cipher' + str(i) for i in range(36)])
columns.extend(['ja3Extension' + str(i) for i in range(26)])
columns.extend(['ja3Ec' + str(i) for i in range(6)])
columns.extend(['ja3Ecpf' + str(i) for i in range(2)])
columns.extend(['blacklisted'])

def normDataFrame(df):
    df = df.drop(df[df['ja3']  == "0"].index , inplace=True)
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


# kolko mame blacklistovych a normalnych instancii
# for root, dirs, files in os.walk('csv_used'):
#     for name in files:
#         c = 0
#         c_b = 0
#         filepath = root + os.sep + name
#         t = pd.read_csv(filepath, header=None)
#         for index, row in t.iterrows():
#             if (t.iloc[index,-1]== 0):
#                 c = c + 1
#             else:
#                 c_b = c_b + 1
#         print(filepath,c,c_b)
# exit()
#y,x = svm_read_problem("csv_used\\2015-03-19_capture-win_normalized.csv")

train2 = pd.read_csv("csv_used\\4SICS-GeekLounge-151021_normalized.csv" , header=None)
train = pd.read_csv("csv_used\\2019-11-12-traffic-analysis-exercise_normalized.csv", header=None)
train1 = pd.read_csv("csv_used\\2021-01-21-traffic-analysis-exercise_normalized.csv", header=None)
test = pd.read_csv("csv_used\\2015-03-19_capture-win_normalized.csv", header=None)

#test = pd.read_csv("csv_used\\4SICS-GeekLounge-151021_normalized.csv", header=None)

x = train.iloc[: , 1:-1]
y = train.iloc[: , -1: ]
x1 = train1.iloc[: , 1:-1]
y1 = train1.iloc[: , -1: ]
x2 = train2.iloc[: , 1:-1]
y2 = train2.iloc[: , -1: ]

x_test = test.iloc[ : , 1:-1]
y_test = test.iloc[: , -1: ]

#svmclf = sklearn.svm.SVC(kernel='linear', C=1, gamma=0.001 ,probability=True)
svmclf = sklearn.svm.SVC(kernel='poly', C=1, gamma=0.001 ,probability=True)
#svmclf = sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=100.0, max_iter=10000)
x = x.append(x1)
#x = x.append(x2)
y = y.append(y1)
#y = y.append(y2)
print(x)
svmclf.fit(x,y)





print("prediction:", svmclf.predict(x_test))
pred = svmclf.predict(x_test)

acc = sklearn.metrics.accuracy_score(y_test, pred)
print(acc)


skplt.metrics.plot_confusion_matrix(y_test, svmclf.predict(x_test), normalize=True)
plt.show()


corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]


# print(x_test)
#
# support_vectors = svmclf.support_vectors_
#
# support_vectors_per_class = svmclf.n_support_
# print(support_vectors_per_class)
# plt.scatter(x_test.squeeze().values.reshape(-1,2)[:,0], x_test.squeeze().values.reshape(-1,2)[:,1])
# plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
# plt.title('Linearly separable data with support vectors')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()
# exit()
#
# colors = ['green']
# for n, color in enumerate(colors):
#     idx = np.where(y_test.squeeze().values.reshape(-1, 1) == n)[0]
#     plt.scatter(x_test.squeeze().values.reshape(-1,1)[idx,0], x_test.squeeze().values.reshape(-1,1)[idx,0], color=color)
# plt.scatter(x_test.squeeze().values.reshape(-1,1)[incorr, 0], x_test.squeeze().values.reshape(-1,1)[incorr,0], color='red')
# plt.show()
#for root, dirs, files in os.walk('csv'):
 #   for name in files:
  #      filepath = root + os.sep + name
   #     normDataFrame( pd.read_csv(filepath, low_memory=False))
    #    print(filepath)

# for root, dirs, files in os.walk('csv_used'):
#     for name in files:
#         filepath = root + os.sep + name
#         if filepath.startswith("csv_used"):
#             train = pd.read_csv(filepath)
#             print(filepath)
#             try:
#                 train=normDataFrame(train)
#             except:
#                 continue
#             min_max_scaler = preprocessing.MinMaxScaler()
#             x_scaled = min_max_scaler.fit_transform(train.values)
#             train = pd.DataFrame(x_scaled, columns=columns)
#             train.to_csv(str(filepath)[:-4]+'_normalized' + '.csv',  header=False)
# exit()

# train = pd.read_csv("csv_used\\2015-03-19_capture-win.csv") # norm:470 bl:85
# test = pd.read_csv("csv\\2021-01-21-traffic-analysis-exercise.csv")
# train = normDataFrame(train)
# test = normDataFrame(test)
# #normalizacia?
#
#
#
#
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(train.values)
# train = pd.DataFrame(x_scaled, columns=columns)
#
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(test.values)
# test = pd.DataFrame(x_scaled, columns=columns)
# testB=test.pop('blacklisted')
# test = test.to_numpy()
# feature = train.columns
# trainBlacklisted = train.pop('blacklisted')
# print(trainBlacklisted.shape)
# train1Black = trainBlacklisted
#
# train= train.to_numpy()
#
# train1 = train
#
# model = getSVM()
# print(model.summary())
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
#
#
# model.fit(x=train, y=trainBlacklisted, validation_split=0.1,batch_size=1, epochs=20, verbose=2, callbacks=[tensorboard])
# #model_graph = tf.keras.utils.model_to_dot(model, expand_nested=True, subgraph=True)
# #graph = pydot.graph_from_dot_data(model_graph)
# #print(model_graph.get_nodes())
# #model_graph.write_raw('model.dot')
# #model_graph.write_png('model.png')
#
#
#
# model1 = get_rm_model()
# print(model1.summary())
# keras.utils.plot_model(model1, "my_2nd_model_with_shape_info.png", show_shapes=True)
# print(train1.shape,train1Black.shape)
# model1.fit(x=train1, y=train1Black, validation_split=0.1,batch_size=1, epochs=20, verbose=1, callbacks=[tensorboard])
#
# testB = testB.to_numpy()
#
# y_0 = model.predict_classes(test)
# y_1 = model1.predict_classes(test)
# model_ac = model1_ac = 0
# false_pos = false_pos1 = 0
# false_neg = false_neg1 = 0
# for i in range(len(test)):
#
#     try:
#         print("X=%d, Predicted=%s Real=%s" % (i, y_0[i], testB[i]))
#
#         if (y_0[i] == 1 and  int(testB[i] == 0)):
#             false_neg = false_neg + 1
#         elif (y_0[i] == 0 and  int(testB[i] == 1)):
#             false_pos = false_pos + 1
#         else:
#             model_ac = model_ac + 1
#         print("X=%d, Predicted=%s Real=%s"% (i, y_1[i], testB[i]))
#
#
#         if (y_1[i] == 1 and  int(testB[i] == 0)):
#             false_neg1 = false_neg1 + 1
#         elif (y_1[i] == 0 and  int(testB[i] == 1)):
#             false_pos1 = false_pos1 + 1
#         else:
#             model1_ac = model1_ac + 1
#     except:
#         pass
#
# print("acc model = ", model_ac/len(test))
# print("acc model = ", model1_ac/len(test))
#
# print("false_pos model = ", false_neg/len(test))
# print("false_pos model = ", false_neg1/len(test))
#
# print("false_neg model = ", false_pos/len(test))
# print("false_neg model = ", false_pos1/len(test))

#score=model.evaluate(norm_test, y_test, verbose=2)
#score1=model1.evaluate(norm_test, y_test, verbose=2)
#model.evaluate(dataset_t)

