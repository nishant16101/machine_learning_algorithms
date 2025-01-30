import numpy as np
class LogisticRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def fit(self,X,y):
        self.m,self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        
        for _ in range(self.epochs):
            z = np.dot(X,self.weights) + self.bias
            y_pred = self.sigmoid(z)
            dw = (1/self.m) * np.dot(X.T,(y_pred-y))
            db = (1/self.m) * np.sum(y_pred-y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,X):
        z = np.dot(X,self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in y_pred]
    
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1]) 

model = LogisticRegression(learning_rate=0.01,epochs=1000)
model.fit(X,y)
X_new = np.array([[1.5],[3.5],[6]])
predictions = model.predict(X_new)
print(predictions)
        