import numpy as np
import matplotlib.pyplot as plt
class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    def fit(self,X):
        #Step1 : Mean Centering
        self.mean = np.mean(X,axis=0)
        X_centrerd = X-self.mean

        #step 2:Covariance matrix
        cov = np.cov(X_centrerd.T)
        #step 3 :Eigen decomposition
        eigenvalues,eigenvectors = np.linalg.eigh(cov)
        
        #step4: Sort the eigenvectors by descending eigenvalue
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:,idxs]
        eigenvalues = eigenvalues[idxs]

        #select top k
        self.components = eigenvectors[:,:self.n_components]
    

    def transform(self,X):
        X_centered = X-self.mean
        return np.dot(X_centered,self.components)
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
# Generate toy 3D dataset
np.random.seed(0)
X = np.dot(np.random.rand(3, 3), np.random.randn(3, 200)).T

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced)

# Plot 2D projected data
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, edgecolor='k')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA — 2D Projection')
plt.grid(True)
plt.show()


