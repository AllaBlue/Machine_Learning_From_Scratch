import numpy as np

class StandardScaler():
    """
    A class that standardizes features by removing the mean and scaling to unit variance.

    The standard score of a sample is calculated as:
        z = (X - mean) / std

    Attributes:
        mean_ (list of float): The mean value for each feature in the training set.
        std_ (list of float): The standard deviation for each feature in the training set.
        z_score_ (numpy.ndarray): The standardized (z-score) values of the dataset.
    """

    def __init__(self) -> None:
        """
        Initializes the StandardScaler with empty lists for mean, standard deviation, and z-score.
        """
        self.mean_ = []
        self.std_ = []
        self.z_score_ = []

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Computes the mean and standard deviation for each feature in the dataset.

        Args:
            X (numpy.ndarray): The input data to fit the scaler on, with shape (n_samples, n_features).
        
        Returns:
            None
        """
        self.mean_ = []
        self.std_ = []
        self.z_score_ = []

        for column in range(X.shape[1]):
            column_mean = np.mean(X[:,column])
            column_std = np.std(X[:,column])

            self.mean_.append(column_mean)
            self.std_.append(column_std)
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardizes the dataset by scaling each feature to zero mean and unit variance.

        Args:
            X (numpy.ndarray): The input data to transform, with shape (n_samples, n_features).
        
        Returns:
            numpy.ndarray: The standardized dataset with shape (n_samples, n_features).
        """
        self.z_score_ = np.array((X - self.mean_) / self.std_, dtype="float64").reshape(X.shape)
        return self.z_score_.reshape(X.shape)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the scaler to the dataset and then transforms it.

        This method combines `fit` and `transform` into a single step.

        Args:
            X (numpy.ndarray): The input data to fit and transform, with shape (n_samples, n_features).
        
        Returns:
            numpy.ndarray: The standardized dataset with shape (n_samples, n_features).
        """
        self.fit(X)
        self.transform(X)
        
        return self.z_score_.reshape(X.shape)
