import numpy as np

class BinaryClassification:
    def __init__(self,learning_rate = 0.01,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y).reshape(-1,1)
        n_samples ,n_features = X.shape
        #intitalize weights
        self.w = np.zeros((n_features,1))
        self.b = 0

        #gradient descent
        for _ in range(self.epochs):
            linear_model = np.dot(X,self.w) +self.b
            y_pred = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T,(y-y_pred   ))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict_proba(self,X):
        X = np.array(X)
        return self.sigmoid(np.dot(X,self.w)+ self.b)
    def predict(self,X,threshold=0.5):
        return (self.predict_proba(X)>=threshold).astype(int)


# Generate simple dataset
np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)  # Simple linear boundary

# Train classifier
clf =BinaryClassification(learning_rate=0.1, epochs=1000)
clf.fit(X_train, y_train)

# Predict on new data
X_test = np.array([[0.5, 0.5], [-1, -1]])
pred_probs = clf.predict_proba(X_test)
pred_labels = clf.predict(X_test)


print("Predicted Probabilities:", pred_probs.flatten())
print("Predicted Labels:", pred_labels.flatten())
