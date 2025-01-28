

## Linear Regression using Gradient descent 

import numpy as np
class LinearRegression:
    def __init__(self,learning_rate =0.01,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs =epochs
    
    def fit(self,X,y):
        self.m ,self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = np.dot(X,self.weights) + self.bias
            dw = -(2/self.m) * np.dot(X.T,(y-y_pred))
            db = -(2/self.m) * np.sum(y-y_pred)
            self.weights -= self.learning_rate *dw
            self.bias -= self.learning_rate * db
        
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
