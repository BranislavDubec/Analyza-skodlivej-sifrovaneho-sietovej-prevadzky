"""
    Script uses 4 tehniques of machine learning to train and test dataset defined in the beginning.
    Each technique is hypertuned by method GridSearch and the best parameters are picked on test datas;
    and creates confusion matrics of each technique that is shown and stored as jpg format.
    Name of the file is scorint parameter and parameters of the estimator.
    Script also display additional information on standard output.
"""
__author__ = "Branislav Dubec"
__credits__ = ["Petr Chmelar"]
__version__ = "1.0.0"



import pandas as pd
import numpy as np
import os
from textwrap import wrap
import sklearn
from sklearn import neighbors
from libsvm.svmutil import *
from sklearn import svm
from sklearn import tree
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import scikitplot as skplt
import time


# dict of accuracies on tested instancies with correspondning estimator
accuracies = {}
# best value of tested instancies
best = 0

# load files to train and test
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

# appends test and train datasets; to have more instancies
x_test = x_test.append(x_test1)
x_test = x_test.append(x_test2)

y_test = y_test.append(y_test1)
y_test = y_test.append(y_test2)

x = x.append(x1)

y = y.append(y1)



"""
    Returns list of best parameters found in function GridSearch().
    Display on standard output additional information.
"""
def bestParamsfromGrid(grid):
    best_std = grid.cv_results_['std_test_score'][grid.best_index_]
    params_dic = []
    for i in range(len(grid.cv_results_['rank_test_score'])):
        if grid.cv_results_['rank_test_score'][i] == 1 and best_std >= grid.cv_results_['std_test_score'][i]:
            params_dic.append(grid.cv_results_['params'][i])


    print("Best parameters set found in grid:")
    print()
    print(grid.best_params_)
    print("Best score found in grid:")
    print(grid.best_score_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']

    #prints all the values in grid
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%f (+/-%f) for %r"
              % (mean, std * 2, params))
    print("Estimators with the same and best accuracy from gridsearch: ", params_dic)
    return params_dic


"""
    Fits classifier and sets variable 'best' to best predicted value on tested instancies.
    Display additional information on standart output.
"""
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


"""
    Cross validate best estimator on tested instancies, creates and shows confusion matrix of the accuracy.
    Display classification report.
    Display additional information on standard output.
"""
def bestPredict(score_alg,clf):
    print("Accuracy on cross valuidation on tested instancies.")
    scores = sklearn.model_selection.cross_validate(clf, x_test, y_test, cv=5, scoring='accuracy',
                                                    return_train_score=True)
    for i, score in enumerate(scores["test_score"]):
        print(f"Accuracy for the fold no. {i} on the test set: {score}")
    print("average score: ", np.mean(scores["test_score"]))
    pred = clf.predict(x_test)

    print("Table of predictions: ", pred)

    skplt.metrics.plot_confusion_matrix(y_test, clf.predict(x_test))

    plt.title("\n".join(wrap("scoring: " + score_alg +  " parameters of classifier: " + str(clf),100)))
    name = str(clf).replace('\n', '')
    name = name.replace('\t', '')
    plt.savefig(score_alg + " " + name + '.jpg', bbox_inches='tight')
    plt.show()

    y_true, y_pred = y_test, clf.predict(x_test)
    print(sklearn.metrics.classification_report(y_true, y_pred, zero_division=True))
    print()

    # corr = np.where(y_test.squeeze().values.reshape(-1, 1) == pred.reshape(-1, 1))[0]
    # incorr = np.where(y_test.squeeze().values.reshape(-1, 1) != pred.reshape(-1, 1))[0]
    # print("correct index : ", corr)
    # print("incorr index: ", incorr)



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




# gamma = 2^-15, 2^-13-..... 2^3
# C =´2^-5.... 2^12
"""
    SVM technique.
    param_grid = values for SVM technique.
    scores = different scoring for gridsearch().
"""
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

scores = ['precision', 'accuracy']





for score in scores:
    best = 0
    accuracies = {}
    start = time.time()
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid = sklearn.model_selection.GridSearchCV(
        sklearn.svm.SVC(), param_grid, scoring='%s' % score, cv=5 , refit=True
    )
    grid.fit(x, y)


    params_dic = bestParamsfromGrid(grid)

    for param in params_dic:
        try:
            clf = svm.SVC()
            clf.set_params(C=param['C'], gamma=param['gamma'], kernel=param['kernel'],probability=True)
        except:
            clf = svm.SVC()
            clf.set_params(C=param['C'], kernel=param['kernel'],probability=True)

        fitPredict(clf)
    print("Best value on tested instancies: ",best)
    print("Accuracies of tested instancies for each estimator in params_dic:")
    print(accuracies)
    end = time.time()
    print("Time in seconds: ", end-start)
    bestPredict(score, accuracies[best])



"""
    K-nearest neighbor technique.
    param_grid = values for k-nearest neighbor technique.
    scores = different scoring for gridsearch().
"""



k_range = list(range(1, 31))

param_grid = {'n_neighbors' : k_range, 'weights' : ['uniform', 'distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'] }

scores = ['precision', 'accuracy']
for score in scores:
    best = 0
    accuracies = {}
    start = time.time()
    print("# Tuning hyper-parameters for %s" % score)
    print()
    grid = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KNeighborsClassifier(), param_grid, cv=10, scoring=score)
    grid.fit(x,y)

    params_dic = bestParamsfromGrid(grid)

    for param in params_dic:
        clf  = sklearn.neighbors.KNeighborsClassifier(n_neighbors = param['n_neighbors'], algorithm = param['algorithm'], weights = param['weights'])
        fitPredict(clf)

    print("Best value on tested instancies: ", best)
    print("Accuracies of tested instancies for each estimator in params_dic:")
    print(accuracies)
    end = time.time()
    print("Time in seconds: ", end - start)
    bestPredict(score,accuracies[best])


"""
    Decision tree technique.
    n_components = different values for PCA function.
    param_grid = values for decision tree technique.
    scores = different scoring for gridsearch().
"""


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
    start = time.time()
    print("# Tuning hyper-parameters for %s" % score)
    grid = sklearn.model_selection.GridSearchCV(pipe, param_grid, scoring=score)
    grid.fit(x,y)
    params_dic = bestParamsfromGrid(grid)




    for param in params_dic:
        dt_clf  = sklearn.tree.DecisionTreeClassifier(criterion = param['dec_tree__criterion'], max_depth = param['dec_tree__max_depth'])
        pca_clf = sklearn.decomposition.PCA(n_components = param['pca__n_components'])
        clf = Pipeline([('pca', pca_clf),
                         ('tree', dt_clf)])
        fitPredict(clf)

    print("Best value on tested instancies: ", best)
    print("Accuracies of tested instancies for each estimator in params_dic:")
    print(accuracies)
    end = time.time()
    print("Time in seconds: ", end - start)
    bestPredict(score,accuracies[best])

"""
    Multi layer perceptron technique.
    param_grid = values for MLP technique.
    scores = different scoring for gridsearch().
"""

mlp = MLPClassifier(max_iter=300)
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
for score in scores:


    best = 0
    accuracies = {}
    start = time.time()
    print("# Tuning hyper-parameters for %s" % score)
    clf = sklearn.model_selection.GridSearchCV(mlp, param_grid, n_jobs=-1, cv=5, scoring=score)
    clf.fit(x, y)
    params_dic = bestParamsfromGrid(clf)

    for param in params_dic:
        clf = MLPClassifier(hidden_layer_sizes=param['hidden_layer_sizes'], activation=param['activation'],
                            solver=param['solver'], alpha=param['alpha'], learning_rate=param['learning_rate'],
                            max_iter=300)

        fitPredict(clf)

    print("Best value on tested instancies: ", best)
    print("Accuracies of tested instancies for each estimator in params_dic:")
    print(accuracies)
    end = time.time()
    print("Time in seconds: ", end - start)
    bestPredict(score, accuracies[best])