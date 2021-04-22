import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

def optimize_lambda(X, y, folds=2):
    assert(len(X) == len(y))
    assert(len(X) > 0)
    lambdas = []
    for i in range(1,100):
        lambdas.append(0.1*i)

    max_lambda = max(lambdas)
    LRs = {}
    accuracies = {}
    ch = int(len(X)//folds)

    for fold in (range(folds)):
        i = range(fold*ch, (fold+1)*ch)
        current_fold = pd.Series([False for i in range(len(X))])
        current_fold.loc[i] = True
        X_train, y_train = X[~current_fold].reset_index(drop=True), y[~current_fold].reset_index(drop=True)
        X_test, y_test = X[current_fold].reset_index(drop=True), y[current_fold].reset_index(drop=True)
        LR = LogisticRegression(fit_intercept=fit_intercept)
        LRs[fold+1] = LR

        for Lambda in lambdas:
            LR.samantha = Lambda
            LR.fit_L1_autograd(X_train, y_train)
            y_hat = LR.predict(X_test)
            if fold+1 in accuracies:
                accuracies[fold+1][Lambda] = accuracy(y_hat, y_test)
            else:
                accuracies[fold+1] = {Lambda: accuracy(y_hat, y_test)}

    accuracies = pd.DataFrame(accuracies).transpose()
    accuracies.index.name = "Fold Number"
    accuracies.loc["mean"] = accuracies.mean()
    print("Optimum lambda = {}".format(accuracies.loc["mean"].idxmax()))

# def optimize_lambda(X,y):
#     LR = LogisticRegression(fit_intercept=True)
#     alpha = 0.01
#     maxlamda = 0
#     maxaccu = 0
#     for i in range (0,100):
#         LR.samantha = alpha*i
#         LR.fit_L1_autograd(pd.DataFrame(X),pd.Series(y))
#         y_hat = LR.predict(pd.DataFrame(X))
#         accu = accuracy(y_hat,y)
#         if accu >= maxaccu :
#             maxaccu = accu
#             maxlamda = alpha*i
#     return maxlamda

np.random.seed(42)
N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series([0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1])
fit_intercept = True
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_L1_autograd(pd.DataFrame(X),pd.Series(y))
y_hat = LR.predict(pd.DataFrame(X))
print ((accuracy(y_hat,y)))
LR.fit_L2_autograd(pd.DataFrame(X),pd.Series(y))
y_hat = LR.predict(pd.DataFrame(X))
print ((accuracy(y_hat,y)))
print(LR.coef_)
optimize_lambda(pd.DataFrame(X),pd.DataFrame(y))

# x = load_breast_cancer(as_frame=True)
# X = x.data
# s = MinMaxScaler()
# X[list(X)]= s.fit_transform(X)
# y = x.target
# X = np.array(X)
# y = np.array(y)
# kf = KFold(n_splits=3)
# a = []
# b = []
# fit_intercept = True
# LR = LogisticRegression(fit_intercept=fit_intercept)

# for train,test in kf.split(X,y):
    
#     LR.fit_L1_autograd(pd.DataFrame(X[train]), pd.Series(y[train]))
#     y_hat = LR.predict(pd.DataFrame(X[test]))
#     a.append(accuracy(y_hat,y[test]))
# print (a)

# for train,test in kf.split(X,y):

#     LR.fit_L2_autograd(pd.DataFrame(X[train]), pd.Series(y[train]))
#     y_hat = LR.predict(pd.DataFrame(X[test]))
#     b.append(accuracy(y_hat,y[test]))
# print (b)