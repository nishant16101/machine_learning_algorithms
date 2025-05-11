import numpy as np
from collections import defaultdict
class GaussianNaiveBayes:
    def fit(self,X,y):
        self.classes = np.unique(y)
        self.priors = {cls:np.mean(y==cls) for cls in self.classes}
        self.mean = {}
        self.var = {}

        for cls in self.classes:
            X_cls = X[y==cls]
            self.mean[cls] = np.mean(X_cls,axis=0)
            self.var[cls] = np.var(X_cls,axis=0)+1e-9
    def _gaussian_pdf(self,x,mean,var):
        numerator = np.exp(-(x-mean)**2)/(2*var)
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator
    def _class_likelihood(self,x,cls):
        probs = self._gaussian_pdf(x,self.mean[cls],self.var[cls])
        return np.prod(probs)*self.priors[cls]
    def predict(self,X):
        y_pred = []
        for x in X:
            posteriors = {cls: self._class_likelihood(x, cls) for cls in self.classes}
            y_pred.append(max(posteriors, key=posteriors.get))
        return np.array(y_pred)
    
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
