import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
class GMM:

    def __init__(self, C, n_runs):
        self.C=C
        self.n_runs=n_runs

    def get_params(self):
        return self.mu, self.pi, self.sigma

    def calculate_mean_covariance(self, X, predictions):

        self.initial_mean= np.zeros((self.C, X.shape[1]))
        self.initial_covariance= np.zeros((self.C, X.shape[1], X.shape[1]))
        self.initial_pi=np.zeros(self.C)

        labels= np.unique(predictions)

        
        for label in labels:
            X_label= X[predictions==label]
            self.initial_mean[label, :]= np.mean(X_label,axis=0)
            self.initial_pi[label]=X_label.shape[0]/ X.shape[0]
            self.initial_covariance[label, :, :]= np.dot((X_label- self.initial_mean[label, :]).T,(X_label- self.initial_mean[label, :]) )
            
        return self.initial_mean, self.initial_pi, self.initial_covariance

    def K_means_initial_params(self, X):
        kmeans= KMeans(n_clusters=self.C, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)
        predictions= kmeans.predict(X)

        self.initial_mean, self.initial_pi, self.initial_covariance=self.calculate_mean_covariance(X, predictions)

        return self.initial_mean, self.initial_pi, self.initial_covariance

    def e_step(self, X):

        self.mu, self.pi, self.sigma = self.mu, self.pi, self.sigma if ((self.mu is not None) and(self.pi is not None) and (self.sigma is not None)) else K_means_initial_params(X)

        self.gamma=np.zeros((X.shapep[0], self.C))

        for c in range(self.C):
            Gauusian_c_pdf= multivariate_normal.pdf(X, mean=self.mu[c, :], cov=self.sigma[c,:,:])
            self.gamma[:, c]= self.pi[c] * Gauusian_c_pdf
        
        self.gamma= self.gamma / np.sum(self.gamma, axis=1).reshape((-1,1))

        return self.gamma





        



