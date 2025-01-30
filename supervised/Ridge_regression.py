import numpy as np
class Ridgeregression:
    def __init__(self,learning_rate=0.01,lamda_=1.0,epochs=1000):
        self.learning_rate = learning_rate
        self.lamda_ = lamda_
        self.epochs = epochs
    def fit(self,X,y):
        self.m , self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X,self.weights) + self.bias
            dw = -(2/self.m) * np.dot(X.T,(y-y_pred)) + 2 * self.lamda_ *self.weights
            db = -(2/self.m) * np.sum(y-y_pred)

            self.weights -= self.learning_rate* dw
            self.bias -= self.learning_rate* db
    
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
    
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = Ridgeregression(learning_rate=0.01,lamda_=1.0,epochs=1000)
model.fit(X,y)

X_new = np.array([[6],[7]])
predictions = model.predict(X_new)

print(predictions)

