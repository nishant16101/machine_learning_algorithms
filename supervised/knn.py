import numpy as np
from collections import Counter
class KNN:
    def __init__(self,k=3,task='classification '):
        self.k = k
        self.task = task
    def fit(self,X_train,y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def _euclidean(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def _predict_point(self,x):
        distances = [self._euclidean(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        if self.task == 'classification':
            return Counter(k_labels).most_common(1)[0][0]
        elif self.task == 'regression':
            return np.mean(k_labels)

    def predict(self,X_test):
        return np.array([self._predict_point(x) for x in X_test]) 


from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)


split = 80
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = KNN(k=5, task='classification')
model.fit(X_train, y_train)
preds = model.predict(X_test)

accuracy = np.mean(preds == y_test)
print(f"Accuracy: {accuracy:.2f}")
       