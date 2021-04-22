import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nets.nn.modules import Module
from nets.nn.activation import *
# from logisticRegression.logisticRegression import LogisticRegression
from NN.NN import DNN
from metrics import *

np.random.seed(42)
N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series([0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1])

NN = DNN(layer_dimensions = [2,30,1])
NN.fit(pd.DataFrame(X),pd.DataFrame(y))
y_hat = NN.predict(pd.DataFrame(X))
print (accuracy(y_hat,y))