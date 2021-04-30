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




# kolko mame blacklistovych a normalnych instancii v suboroch
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





train = pd.read_csv("csv_used\\2019-11-12-traffic-analysis-exercise_normalized.csv", header=None)
train1 = pd.read_csv("csv_used\\2021-01-21-traffic-analysis-exercise_normalized.csv", header=None)
test = pd.read_csv("csv_used\\2015-03-19_capture-win_normalized.csv", header=None)



x = train.iloc[: , 1:-1]
y = train.iloc[: , -1: ]
x1 = train1.iloc[: , 1:-1]
y1 = train1.iloc[: , -1: ]


x_test = test.iloc[ : , 1:-1]
y_test = test.iloc[: , -1: ]

#svmclf = sklearn.svm.SVC(kernel='linear', C=1, gamma=0.001 ,probability=True)
#svmclf = sklearn.svm.SVC(kernel='poly', C=1, gamma=0.001 ,probability=True)
svmclf = sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=100.0, max_iter=10000)
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
print(corr)


