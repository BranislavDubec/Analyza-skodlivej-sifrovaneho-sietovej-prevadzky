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
from sklearn import neighbors
from libsvm.svmutil import *
from sklearn import svm
from sklearn import tree
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
import scikitplot as skplt
import time


# dict of accuracies on tested instancies with correspondning estimator
accuracies = {}
# best value of tested instancies
best = 0

# load files to train and test
train = pd.read_csv("csv_used\\normalized\\2019-11-12-traffic-analysis-exercise_normalized.csv", header=None)
train = pd.read_csv("csv_used\\normalized\\2021-01-21-traffic-analysis-exercise_normalized.csv", header=None)
test = pd.read_csv("csv_used\\normalized\\2015-03-19_capture-win_normalized.csv", header=None)
test1 = pd.read_csv("csv_used\\normalized\\2021-02-08-traffic-analysis-exercise_normalized.csv ", header=None)
test2 = pd.read_csv("csv_used\\normalized\\2020-02-21-traffic-analysis-exercise_normalized.csv", header=None)
train1 = pd.read_csv("csv_used\\normalized\\2018-12-18-traffic-analysis-exercise_normalized.csv", header=None)


# separates features and labels
x = train.iloc[: , :-1]
y = train.iloc[: , -1: ]
x1 = train1.iloc[: , :-1]
y1 = train1.iloc[: , -1: ]


x_test = test.iloc[ : , :-1]
y_test = test.iloc[: , -1: ]

x_test1 = test1.iloc[ : , :-1]
y_test1 = test1.iloc[: , -1: ]

x_test2 = test2.iloc[ : , :-1]
y_test2 = test2.iloc[: , -1: ]

x_test = x_test.append(x_test1)
x_test = x_test.append(x_test2)

y_test = y_test.append(y_test1)
y_test = y_test.append(y_test2)
#svmclf = sklearn.svm.SVC(kernel='rbf', C=100, gamma=0.001 ,probability=True)
x = x.append(x1)

y = y.append(y1)



def bestParamsfromGrid(grid):
    best_std = grid.cv_results_['std_test_score'][grid.best_index_]
    params_dic = []
    for i in range(len(grid.cv_results_['rank_test_score'])):
        if grid.cv_results_['rank_test_score'][i] == 1 and best_std >= grid.cv_results_['std_test_score'][i]:
            params_dic.append(grid.cv_results_['params'][i])
    ## Viacero tuningov ma rovnake hodnotenie  cross selection na najdenie najlepsieho

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print(grid.best_score_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%f (+/-%f) for %r"
              % (mean, std * 2, params))
    print()
    return params_dic

def fitPredict(clf):
    global best,accuracies
    print("Classifier:", clf)
    print("Fitting")
    clf.fit(x, y)
    pred_train = clf.predict(x)
    acc_train = sklearn.metrics.accuracy_score(y, pred_train)
    print("Accuracy on the whole train dataset:", acc_train)

    pred = clf.predict(x_test)

    acc = sklearn.metrics.accuracy_score(y_test, pred)
    print("Accuracy of tested instancies:", acc)

    if (best >= acc and acc in accuracies.keys()):
        print("Accuracy is worse that we have.")
        return

    accuracies[acc] = clf
    if best < acc:
        best = acc

def bestPredict(score_alg,clf):
    print("Accuracy on cross valuidation on tested instancies.")
    scores = sklearn.model_selection.cross_validate(clf, x_test, y_test, cv=7, scoring='accuracy',
                                                    return_train_score=True)
    for i, score in enumerate(scores["test_score"]):
        print(f"Accuracy for the fold no. {i} on the test set: {score}")
    print("avr score: ", np.mean(scores["test_score"]))
    pred = clf.predict(x_test)
    print("Table of predictions: ", pred)

    skplt.metrics.plot_confusion_matrix(y_test, clf.predict(x_test))
    plt.title(" score: " + score_alg +  " params: " + str(clf))
    plt.show()

    y_true, y_pred = y_test, clf.predict(x_test)
    print(sklearn.metrics.classification_report(y_true, y_pred, zero_division=True))
    print()

    corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
    incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]
    print("corr index: ", corr)
    print("incorr index: ", incorr)
    prob = clf.predict_proba(x_test)
    print("Predict probability table: ", prob)
    for i in incorr:
        print(i, " ", prob[i])


# number of good and bad instancies in datasets
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







# param_grid  = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                      'C': [0.1, 1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
#                     {'C': [0.1, 1, 10, 100, 1000],
#                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                      'kernel': ['sigmoid']}
#                     ]
# gamma = 2^-15, 2^-13-..... 2^3
# C =´2^-5.... 2^12
param_grid  = [{'kernel': ['rbf'], 'gamma': [0.00003051757,0.00006103515, 0.00012207031, 0.00024414062,
                                             0.00048828125, 0.0009765625, 0.001953125,
                                             0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8],
                     'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
                    {'kernel': ['linear'], 'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
                    {'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                     'gamma': [0.00003051757,0.00006103515, 0.00012207031, 0.00024414062,
                                             0.00048828125, 0.0009765625, 0.001953125,
                                             0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8],
                     'kernel': ['sigmoid']}
                    ]
# gamma = 2^-15, 2^-13-..... 2^3
# C =´2^6,2^6+16.... 2^8
# param_grid  = [{'kernel': ['rbf'], 'gamma': [ 0.001953125,0.00211588541, 0.00227864582, 0.00244140623,0.00260416664,0.00276692705,0.00292968746,
#                                                 0.00309244787,0.00325520828,0.00341796869,0.0035807291,0.00374348951,
#                                              0.00390625, 0.0078125],
#                      'C': [ 64,80,96, 128,144,160,176,190,206,222,238, 256],'max_iter': [1000,100000]},
#                     {'kernel': ['linear'], 'C': [ 64,80,96, 128,144,160,176,190,206,222,238, 256]},
#                     {'C': [ 64,80,96, 128,144,160,176,190,206,222,238, 256],
#                       'gamma': [ 0.001953125,0.00211588541, 0.00227864582, 0.00244140623,0.00260416664,0.00276692705,0.00292968746,
#                                                 0.00309244787,0.00325520828,0.00341796869,0.0035807291,0.00374348951,
#                                              0.00390625, 0.0078125],'max_iter': [1000,100000],
#                      'kernel': ['sigmoid']}
#                     ]
scores = ['precision', 'accuracy']



#
for score in scores:
    best = 0
    accuracies = {}
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid = sklearn.model_selection.GridSearchCV(
        sklearn.svm.SVC(), param_grid, scoring='%s' % score, cv=5 , refit=True
    )
    grid.fit(x, y)


    params_dic = bestParamsfromGrid(grid)
    print("Same gridsearch: ", params_dic)





    for param in params_dic:
        try:
            clf = svm.SVC()
            clf.set_params(C=param['C'], gamma=param['gamma'], kernel=param['kernel'],probability=True)
        except:
            clf = svm.SVC()
            clf.set_params(C=param['C'], kernel=param['kernel'],probability=True)

        # scores = sklearn.model_selection.cross_validate(clf, x, y, cv=6, scoring='accuracy', return_train_score=True)
        # for i, score in enumerate(scores["test_score"]):
        #     print(f"Accuracy for the fold no. {i} on the train  set: {score}")
        # avg_score = np.mean(scores["test_score"])
        # print(avg_score)

        fitPredict(clf)
    print("Best value: ",best)
    print(accuracies)
    bestPredict(score, accuracies[best])



#Knn



k_range = list(range(1, 31))

param_grid = {'n_neighbors' : k_range, 'weights' : ['uniform', 'distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'] }

scores = ['precision', 'accuracy']
for score in scores:
    best = 0
    accuracies = {}
    print("KNN tryout for ", score)
    grid = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KNeighborsClassifier(), param_grid, cv=10, scoring=score)
    grid.fit(x,y)

    params_dic = bestParamsfromGrid(grid)
    print("Same gridsearch: ", params_dic)

    for param in params_dic:
        clf  = sklearn.neighbors.KNeighborsClassifier(n_neighbors = param['n_neighbors'], algorithm = param['algorithm'], weights = param['weights'])
        fitPredict(clf)

    print("Best value: ", best)
    print(accuracies)
    bestPredict(score,accuracies[best])



pipe = Pipeline(steps=[
                           ('pca', sklearn.decomposition.PCA()),
                           ('dec_tree', tree.DecisionTreeClassifier())])

n_components = list(range(1,x.shape[1]+1,1))
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]

param_grid = {
    'pca__n_components' : n_components,
    'dec_tree__criterion' : criterion, 'dec_tree__max_depth' : max_depth,
}

for score in scores:
    best = 0
    accuracies = {}
    print("Dtree tryout for ", score)
    grid = sklearn.model_selection.GridSearchCV(pipe, param_grid, scoring=score)
    grid.fit(x,y)
    params_dic = bestParamsfromGrid(grid)
    print("Same gridsearch: ", params_dic)



    for param in params_dic:
        dt_clf  = sklearn.tree.DecisionTreeClassifier(criterion = param['dec_tree__criterion'], max_depth = param['dec_tree__max_depth'])
        pca_clf = sklearn.decomposition.PCA(n_components = param['pca__n_components'])
        clf = Pipeline([('pca', pca_clf),
                         ('tree', dt_clf)])
        fitPredict(clf)


    print("Best value: ", best)
    print(accuracies)
    bestPredict(score,accuracies[best])