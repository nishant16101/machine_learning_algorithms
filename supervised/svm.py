import numpy as np
class SVM:
    def __init__(self,learning_rate=0.01,lambda_param=0.01,epochs =1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        y = np.where(y==0,-1,1)
        n_samples,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b =0

        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1 # functional margin

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)  # Only regularization
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X[i], y[i]))  # Full gradient
                    self.b -= self.lr * y[i]
    def predict(self,X):
        return np.sign(np.dot(X,self.w) +self.b)
    

np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int) #linear separable
print("Before conversion:", np.unique(y_train))  

svm = SVM(learning_rate=0.01, lambda_param=0.01, epochs=1000)
svm.fit(X_train, y_train)

X_test = np.array([[0.5, 0.5], [-1, -1]])
pred_labels = svm.predict(X_test)

print("Predicted Labels:", pred_labels)