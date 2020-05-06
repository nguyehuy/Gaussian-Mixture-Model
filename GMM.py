import numpy as np
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

        counter=0
        for label in labels:
            X_label= X[predictions==label]
            self.initial_mean[counter, :]= np.mean(X_label,axis=0)
            self.initial_pi[counter]=X_label.shape[1]/ X.shape[1]
            self.initial_covariance= np.dot((X_label- self.initial_mean[counter, :]).T,(X_label- self.initial_mean[counter, :]) )
            counter+=1
        return self.initial_mean, self.initial_pi, self.initial_covariance


