import numpy as np

class StandardScaler():
    def __init__(self):
        self.mean_ = []
        self.std_ = []
        self.z_score_ = []

    def fit(self, X):
        self.mean_ = []
        self.std_ = []
        self.z_score_ = []

        for column in range(X.shape[1]):
            column_mean = np.mean(X[:,column])
            column_std = np.std(X[:,column])

            self.mean_.append(column_mean)
            self.std_.append(column_std)

    def transform(self, X):
        self.z_score_ = np.array((X - self.mean_) / self.std_, dtype="float64").reshape(X.shape)
        return self.z_score_.reshape(X.shape)
    
    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)
        
        return self.z_score_.reshape(X.shape)
