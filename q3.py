import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


x = load_digits(as_frame=True)
X = x.data
s = MinMaxScaler()
X[list(X)]= s.fit_transform(X)
y = x.target
X = np.array(X)
y = np.array(y)
kf = KFold(n_splits=4)
a = []

for train,test in kf.split(X,y):
    batch_size = 5
    fit_intercept = False
    LR = LogisticRegression(fit_intercept=fit_intercept)
    LR.fit_k_class_regularised(pd.DataFrame(X),pd.Series(y),n_iter=500)
    y_hat = LR.k_class_predict(pd.DataFrame(X),False)
    a.append(accuracy(y_hat,y))
print (a)

X  = np.array([[1.0,2.0],[2.0,1.0],[3.0,1.0],[8.0,9.0],[9.0,8.0]])
y  = np.array([0,0,0,1,1])
fit_intercept = False
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_k_class_regularised(pd.DataFrame(X),pd.Series(y),n_iter=500)
y_hat = LR.k_class_predict(pd.DataFrame(X),False)
# print(y_hat)
# print("y",y)
print ((accuracy(y_hat,y)))
print (LR.confusion(X,y))

# #Autograd
# X  = np.array([[1.0,2.0],[2.0,1.0],[3.0,1.0],[8.0,9.0],[9.0,8.0]])
# y  = np.array([0,0,0,1,1])
# fit_intercept = False
# LR = LogisticRegression(fit_intercept=fit_intercept)
# LR.fit_k_class_regularised_autograd(pd.DataFrame(X),pd.Series(y),n_iter=500)
# y_hat = LR.k_class_predict(pd.DataFrame(X),False)
# print ((accuracy(y_hat,y)))
