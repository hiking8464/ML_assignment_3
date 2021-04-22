from nets.nn.modules import Module
from nets.nn.activation import *
import numpy as np
import pandas as pd


class DNN(Module):

    def __init__(self, layer_dimensions, activation_hidden=ReLU()):
        super().__init__()
        self.layer_dimensions = layer_dimensions
        self.hidden_dimensions = layer_dimensions[1: -1]
        assert isinstance(activation_hidden,Activation), "unrecognized activation function type"
        self.activation_hidden = activation_hidden
        self.activation_output = Softmax(axis=1)
        self._parameters = dict()
        self.init_parameters()

    def init_parameters(self):
        for i in range(1, len(self.layer_dimensions)):
            mu = 0
            var = 2 / self.layer_dimensions[i]
            sigma = np.sqrt(var)
            weight_shape = (
                self.layer_dimensions[i - 1], self.layer_dimensions[i])
            weight = np.random.normal(loc=mu, scale=sigma, size=weight_shape)
            bias = np.zeros((self.layer_dimensions[i], ))
            layer_weight = "w_" + str(i)
            self._parameters[layer_weight] = weight
            layer_b = "b_" + str(i)
            self._parameters[layer_b] = bias

    def forward(self, inputs):
        depth = len(self.layer_dimensions) - 1
        z = inputs
        if self.training:
            layer_a = "a_0"
            self._cache[layer_a] = z
        for i in range(1, depth + 1):
            layer_w = "w_" + str(i)
            layer_b = "b_" + str(i)
            weight = self._parameters[layer_w]
            bias = self._parameters[layer_b]
            z = np.dot(z, weight) + bias
            if self.training:
                layer_z = "z_" + str(i)
                self._cache[layer_z] = z
            if i < depth:
                z = self.activation_hidden(z)
                if self.training:
                    layer_a = "a_" + str(i)
                    self._cache[layer_a] = z
        outputs = self.activation_output(z)
        if self.training:
            layer_a = "a_" + str(depth)
            self._cache[layer_a] = outputs

        return outputs

    def backward(self, outputs, labels):
        depth = len(self.layer_dimensions) - 1
        batch_size, num_classes = outputs.shape
        coefficient = 1 / batch_size
        layer_a = "a_" + str(depth - 1)
        a = self._cache[layer_a]
        Jz = outputs - labels
        dw = coefficient * np.dot(a.T, Jz)
        db = coefficient * np.sum(Jz, axis=0)
        self._grad["dw_" + str(depth)] = dw
        self._grad["db_" + str(depth)] = db
        for i in range(depth - 1, 0, -1):
            layer_w = "w_" + str(i + 1)
            layer_a = "a_" + str(i - 1)
            layer_z = "z_" + str(i)
            w = self._parameters[layer_w]
            a = self._cache[layer_a]
            z = self._cache[layer_z]
            Jz = self.activation_hidden.backward(z) * np.dot(Jz, w.T)
            db = coefficient * np.sum(Jz, axis=0)
            dw = coefficient * np.dot(a.T, Jz)
            self._grad["dw_" + str(i)] = dw
            self._grad["db_" + str(i)] = db

    def fit(self, X, y,batch_size = None, n_iter=200):
        

        X = np.array(X)
        y = np.array(y)
        for i in range(1,n_iter):
            for j in range(len(X)):
                hh = self.forward(X[j])
                self.backward(hh,y)
        pass

    def predict(self, X): 
        a = []
        for i in range(len(X)):
            a.append(self.forward(X[i]))

        return a
        pass

