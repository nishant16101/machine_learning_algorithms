import numpy as np
class KernelRegression:
    def __init__(self,kernel='gaussian',bandwidth = 1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def gaussian_kernel(self,x,xi):
        return np.exp(-0.5 * ((x - xi) / self.bandwidth) ** 2) / (self.bandwidth * np.sqrt(2 * np.pi))
    def fit(self,X,y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self,X):
        X = np.array(X)
        y_pred = []
        for x in X:
            weights =np.array([self.gaussian_kernel(x,xi) for xi in self.X_train])
            y_hat = np.sum(weights * self.y_train)/np.sum(weights)
            y_pred.append(y_hat)
        return np.array(y_pred)
    

X_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10]) 

model = KernelRegression(bandwidth=0.5)
model.fit(X_train, y_train)

X_test = np.array([1.5, 2.5, 3.5])
y_pred = model.predict(X_test)

print(y_pred)


    
        