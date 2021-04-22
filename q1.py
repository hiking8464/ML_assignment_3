import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

x = load_breast_cancer(as_frame=True)
X = x.data
s = MinMaxScaler()
X[list(X)]= s.fit_transform(X)
y = x.target
X = np.array(X)
y = np.array(y)
kf = KFold(n_splits=3)
a = []
b = []
for train,test in kf.split(X,y):
    batch_size = 5
    fit_intercept = True
    LR = LogisticRegression(fit_intercept=fit_intercept)
    LR.fit_regularised(pd.DataFrame(X[train]), pd.Series(y[train]),batch_size)
    y_hat = LR.predict(pd.DataFrame(X[test]))
    a.append(accuracy(y_hat,y[test]))
print (a)

for train,test in kf.split(X,y):
    batch_size = 5
    fit_intercept = True
    LR = LogisticRegression(fit_intercept=fit_intercept)
    LR.fit_regularised_autograd(pd.DataFrame(X[train]), pd.Series(y[train]), batch_size)
    y_hat = LR.predict(pd.DataFrame(X[test]))
    b.append(accuracy(y_hat,y[test]))
print (b)

X  = np.array([[1.0,2.0],[2.0,1.0],[3.0,1.0],[8.0,9.0],[9.0,8.0]])
y  = np.array([0,0,0,1,1])
fit_intercept = True
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_regularised(pd.DataFrame(X),pd.Series(y))
y_hat = LR.predict(pd.DataFrame(X))
print ((accuracy(y_hat,y)))
LR.plot(np.array(X),np.array(y))


# np.random.seed(42)
# N = 30
# P = 2
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series([0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1])


