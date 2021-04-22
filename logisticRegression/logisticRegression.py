from autograd.numpy import exp,log
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd.numpy as np 
from autograd import grad 
import math

# def gradient(theta,X,y):
#         X = np.array(X)
#         y = np.array(y)
#         theta = np.array(theta)
#         print("grad",theta)
#         return (y*(log((1.0/(1.0 + exp(-(X.dot(theta))))))) +  (1 - y)*(log(1 - (1.0/(1.0 + exp(-(X.dot(theta))))))))

class LogisticRegression():

    def softmax(self,X):
        t = []
        for i in range(len(self.coef_)):
            t.append(exp(np.dot(X,self.coef_[i])))
        t = np.array(t)
        return t/np.sum(t)

    def confusion(self,X,y):
        k = len(self.coef_)
        m = np.zeros((k,k))
        y_hat = self.k_class_predict(X,False)
        for i in range(len(y_hat)):
            m[y_hat[i],y[i]] += 1
        return m 

    def loss(self,w, X, y):
        n = float(len(X))
        y_pred = np.dot(X, w)
        return -((np.sum(-(y_pred * y) + y*log(1.0 + exp(y_pred))) / n ))

    def loss1(self,w, X, y):
        n = float(len(X))
        samantha = self.samantha
        y_pred = np.dot(X, w)
        return -((np.sum(-(y_pred * y) + log(1.0 + exp(y_pred))) / n) + samantha*(np.sum(np.abs(w))))

    def loss2(self,w, X, y):
        n = float(len(X))
        samantha = self.samantha
        y_pred = np.dot(X, w)
        return -((np.sum(-(y_pred * y) + log(1.0 + exp(y_pred))) / n ) + samantha*(np.sum(np.dot(w.T,w))))

    def kclass(self,X,y,current,soft):
        return (np.sum(X*(np.tile(np.where(y==j,1,0) - soft[:,j],(len(current[0]),1)).T),axis=0))

    def sig(self,x):
        return (1/(1 + exp(-x)))

    def __init__(self, fit_intercept=True):
        
        self.samantha = 0
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass

    def fit_regularised(self, X, y,batch_size = None, n_iter=200, lr=0.01, lr_type='constant'):
       
        if batch_size==None:
            batch_size=len(X)
        self.batch_size=batch_size
        self.n_iter = n_iter
        n = len(X)
        batch_size = len(X)
        temp = lr 
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        column_length = len(X.columns)
        theta = np.zeros(column_length)
        for i in range(1,n_iter):
            current =  theta.copy()
            theta = current - lr*((X.T).dot((self.sig(X.dot(current))) - y))       
        self.coef_ = theta
        pass

    def fit_regularised_autograd(self, X, y, batch_size=None, n_iter=200, lr=0.01, lr_type='constant'):

        self.n_iter = n_iter
        n = len(X)
        temp = lr 
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        column_length = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        theta = np.zeros(column_length)
        gradient = grad(self.loss)
        for i in range(1,n_iter):
            current =  theta.copy()
            theta = current + lr*(gradient(current,X,y))       
        self.coef_ = theta
        pass
        
    def fit_L1_autograd(self, X, y, n_iter=200, lr=0.01, lr_type='constant'):

        self.n_iter = n_iter
        n = len(X)
        temp = lr 
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        column_length = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        theta = np.zeros(column_length)
        L11 = grad(self.loss1)
        for i in range(1,n_iter):
            current =  theta.copy()
            theta = current + lr*(L11(current,X,y))       
        self.coef_ = theta
        pass

    def fit_L2_autograd(self, X, y, n_iter=200, lr=0.01, lr_type='constant'):

        self.n_iter = n_iter
        n = len(X)
        temp = lr 
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        column_length = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        theta = np.zeros(column_length)
        L12 = grad(self.loss2)
        for i in range(1,n_iter):
            current =  theta.copy()
            theta = current + lr*(L12(current,X,y))       
        self.coef_ = theta
        pass

    def fit_k_class_regularised(self, X, y,batch_size = None, n_iter=200, lr=0.01, lr_type='constant'):
       
        if batch_size==None:
            batch_size=len(X)
        self.batch_size=batch_size
        self.n_iter = n_iter
        k = len(np.unique(y))
        n = len(X)
        batch_size = len(X)
        temp = lr 
        column_length = len(X.columns)
        theta = np.zeros((k,column_length))
        self.coef_ = theta
        soft = self.k_class_predict(X)
        for i in range(1,n_iter):
            current =  theta.copy()
            for j in range(k):
                theta[j] = current[j] + lr*(np.sum(X*(np.tile(np.where(y==j,1,0) - soft[:,j],(len(current[0]),1)).T),axis=0))
                   
        self.coef_ = theta
        pass

    def fit_k_class_regularised_autograd(self, X, y,batch_size = None, n_iter=200, lr=0.01, lr_type='constant'):
       
        if batch_size==None:
            batch_size=len(X)
        self.batch_size=batch_size
        self.n_iter = n_iter
        k = len(np.unique(y))
        n = len(X)
        batch_size = len(X)
        temp = lr 
        column_length = len(X.columns)
        theta = np.zeros((k,column_length))
        self.coef_ = theta
        soft = self.k_class_predict(X)
        kclass = grad(self.kclass)
        for i in range(1,n_iter):
            current =  theta.copy()
            for j in range(k):
                theta[j] = current[j] + lr*kclass(X,y,theta,soft)
                   
        self.coef_ = theta
        pass

    def predict(self, X): 
        X_hat = X.copy()
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X_hat))]))
            X_hat = pd.concat([bias,X_hat],axis=1) 
        k = self.sig(np.dot(X_hat,self.coef_))
        p = np.where(k<0.5,0,1)
        return pd.Series(p)
        pass

    def k_class_predict(self, X,flag = True): 
        X = np.array(X)
        soft=np.vectorize(self.softmax,signature='(n)->(m)')
        r = soft(X)
        if flag == True:
            return r
        else:
            return np.argmax(r,axis = 1)
        pass

    def plot(self,X,y):
        b = self.coef_[0]
        w1, w2 = self.coef_[1:]
        c = -b/w2
        m = -w1/w2
        xmin, xmax = -3, 15
        ymin, ymax = -3, 15
        xd = np.array([xmin, xmax])
        yd = m*xd + c
        plt.figure()
        plt.plot(xd, yd, 'k', lw=1, ls='--')
        plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
        plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
        plt.scatter(X[y==0][:,0],X[y==0][:,1])
        plt.scatter(X[y==1][:,0],X[y==1][:,1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')
        plt.savefig('harsha' + '_plot.png')
        plt.show()
        
        pass
