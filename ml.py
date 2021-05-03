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





train = pd.read_csv("csv_used\\normalized\\2019-11-12-traffic-analysis-exercise_normalized.csv", header=None)
train1 = pd.read_csv("csv_used\\normalized\\2021-01-21-traffic-analysis-exercise_normalized.csv", header=None)
test = pd.read_csv("csv_used\\normalized\\2015-03-19_capture-win_normalized.csv", header=None)
#train1 = pd.read_csv("csv_used\\normalized\\2021-02-08-traffic-analysis-exercise_normalized.csv ", header=None)
#train = pd.read_csv("csv_used\\normalized\\2020-02-21-traffic-analysis-exercise_normalized.csv", header=None)
#train = pd.read_csv("csv_used\\normalized\\2018-12-18-traffic-analysis-exercise_normalized.csv", header=None)
x = train.iloc[: , 1:-1]
y = train.iloc[: , -1: ]
x1 = train1.iloc[: , 1:-1]
y1 = train1.iloc[: , -1: ]


x_test = test.iloc[ : , 1:-1]
y_test = test.iloc[: , -1: ]

#svmclf = sklearn.svm.SVC(kernel='rbf', C=100, gamma=0.001 ,probability=True)
svmclf = sklearn.svm.SVC(kernel='sigmoid', C=1000, gamma=0.001 ,probability=True)
#svmclf = sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=100.0, max_iter=10000)
x = x.append(x1)

y = y.append(y1)

# scores = sklearn.model_selection.cross_validate(svmclf, x, y,  cv=5, scoring='accuracy', return_train_score=True)
# for i, score in enumerate(scores["test_score"]):
#     print(f"Accuracy for the fold no. {i} on the test set: {score}")
# avg_score = np.mean(scores["test_score"])
# print(avg_score)
#
# svmclf.fit(x,y)
#
#
#
#
#
#
# print("prediction:", svmclf.predict(x_test))
#
# pred = svmclf.predict(x_test)
#
# acc = sklearn.metrics.accuracy_score(y_test, pred)
# print(acc)
#
#
# skplt.metrics.plot_confusion_matrix(y_test, svmclf.predict(x_test))
# plt.show()
# print(sklearn.metrics.classification_report(y_test, pred, zero_division=True))
#
# corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
# incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]
# print("corr index: ",corr)
# print("incorr index: ", incorr)


# param_grid  = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                      'C': [0.1, 1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
#                     {'C': [0.1, 1, 10, 100, 1000],
#                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                      'kernel': ['sigmoid']}
#                     ]
# gamma = 2^-15, 2^-13-..... 2^3
# C =´2^-5.... 2^12
# param_grid  = [{'kernel': ['rbf'], 'gamma': [0.00003051757,0.00006103515, 0.00012207031, 0.00024414062,
#                                              0.00048828125, 0.0009765625, 0.001953125,
#                                              0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8],
#                      'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
#                     {'kernel': ['linear'], 'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
#                     {'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
#                      'gamma': [0.00003051757,0.00006103515, 0.00012207031, 0.00024414062,
#                                              0.00048828125, 0.0009765625, 0.001953125,
#                                              0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8],
#                      'kernel': ['sigmoid']}
#                     ]ň
# gamma = 2^-15, 2^-13-..... 2^3
# C =´2^6,2^6+16.... 2^8
param_grid  = [{'kernel': ['rbf'], 'gamma': [ 0.001953125,0.00211588541, 0.00227864582, 0.00244140623,0.00260416664,0.00276692705,0.00292968746,
                                                0.00309244787,0.00325520828,0.00341796869,0.0035807291,0.00374348951,
                                             0.00390625, 0.0078125],
                     'C': [ 64,80,96, 128,144,160,176,190,206,222,238, 256]},
                    {'kernel': ['linear'], 'C': [ 64,80,96, 128,144,160,176,190,206,222,238, 256]},
                    {'C': [ 64,80,96, 128,144,160,176,190,206,222,238, 256],
                      'gamma': [ 0.001953125,0.00211588541, 0.00227864582, 0.00244140623,0.00260416664,0.00276692705,0.00292968746,
                                                0.00309244787,0.00325520828,0.00341796869,0.0035807291,0.00374348951,
                                             0.00390625, 0.0078125],
                     'kernel': ['sigmoid']}
                    ]
scores = ['precision', 'accuracy']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = sklearn.model_selection.GridSearchCV(
        sklearn.svm.SVC(), param_grid, scoring='%s' % score, cv=5
    )
    clf.fit(x, y)



    best_std = clf.cv_results_['std_test_score'][clf.best_index_]
    params_dic = []
    for i in range(len(clf.cv_results_['rank_test_score'])):
        if clf.cv_results_['rank_test_score'][i] == 1 and best_std >= clf.cv_results_['std_test_score'][i]:
            params_dic.append(clf.cv_results_['params'][i])
    ## Viacero tuningov ma rovnake hodnotenie  cross selection na najdenie najlepsieho



    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%f (+/-%f) for %r"
              % (mean, std * 2, params))
    print()

    avg_score_best = 0
    clf_used = svm.SVC()
    best = 0
    accuracies = {}
    print("Same gridsearch: ",params_dic)
    for param in params_dic:
        try:
            clf = svm.SVC()
            clf.set_params(C=param['C'], gamma=param['gamma'], kernel=param['kernel'])
        except:
            clf = svm.SVC()
            clf.set_params(C=param['C'], kernel=param['kernel'])
        print(clf)
        scores = sklearn.model_selection.cross_validate(clf, x, y, cv=5, scoring='accuracy', return_train_score=True)
        for i, score in enumerate(scores["test_score"]):
            print(f"Accuracy for the fold no. {i} on the test set: {score}")
        avg_score = np.mean(scores["test_score"])
        print(avg_score)
        print("Fitting")
        clf.fit(x,y)

        if avg_score_best <= avg_score :
            pred = clf.predict(x_test)
            acc = sklearn.metrics.accuracy_score(y_test, pred)
            print("Accuracy of tested instancies:", acc)
            if (best >= acc and acc  in accuracies.keys()):
                print("Accuracy is worse that we have.")
                continue
            accuracies[acc] = clf

            avg_score_best = avg_score
            print("Best score is: ", avg_score_best)
            print("Classifier:", clf)




            print("Table of predictions: ", pred)
            if best < acc:
                best = acc
            skplt.metrics.plot_confusion_matrix(y_test, clf.predict(x_test))
            plt.show()
            y_true, y_pred = y_test, clf.predict(x_test)
            print(sklearn.metrics.classification_report(y_true, y_pred, zero_division=True))
            print()
            corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
            incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]
            print("corr index: ",corr)
            print("incorr index: ", incorr)

    print(accuracies)

# grid = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), param_grid, refit = True)
# grid.fit(x,y)
#
# # print best parameter after tuning
# print(grid.best_params_)
#
# # print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)
#
# pred  = grid.predict(x_test)
#
# acc = sklearn.metrics.accuracy_score(y_test, pred)
# print(acc)
#
#
# skplt.metrics.plot_confusion_matrix(y_test, grid.predict(x_test))
# plt.show()
# print(sklearn.metrics.classification_report(y_test, pred, zero_division=True))
#
# corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
# incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]
# print("corr index: ",corr)
# print("incorr index: ", incorr)
#
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['sigmoid']}
# grid = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), param_grid, refit = True)
#
# grid.fit(x,y)
#
# # print best parameter after tuning
# print(grid.best_params_)
#
# # print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)
#
# pred  = grid.predict(x_test)
#
# acc = sklearn.metrics.accuracy_score(y_test, pred)
# print(acc)
#
#
# skplt.metrics.plot_confusion_matrix( grid.predict(x_test),y_test)
# plt.show()
# print(sklearn.metrics.classification_report(y_test, pred, zero_division=True))
#
# corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
# incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]
# print("corr index: ",corr)
# print("incorr index: ", incorr)
