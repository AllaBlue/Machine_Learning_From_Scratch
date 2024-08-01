import numpy as np

class LinearRegression():
    """
    Ordinary least squares Linear Regression.
    
    Attributes:
        weights (numpy.ndarray): The weights vector including the bias term.
        coef_ (numpy.ndarray): The weights vector without the bias term.
        intercept_ (numpy.ndarray): The bias term.
    """

    def __init__(self) -> None:
        """
        Initializes the LinearRegression model - Ordinary least squares Linear Regression.
        """
        self.weights = []
        self.coef_ = []
        self.intercept_ = 0


    def fit(self, X_train, y_train):
        """
        Fits the Linear Regression model to the training data.
        
        This method calculates the weights vector using the ordinary least squares method:
        
            W = (X.T * X)^(-1) * (X.T * y)
        
        Args:
            X_train (numpy.ndarray): The training data features.
            y_train (numpy.ndarray): The training data target values.
        """
        X = np.hstack((np.ones(X_train.shape), X_train))
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y_train))
        
        self.intercept_ = self.weights[0]
        self.coef_ = self.weights[1:]


    def predict(self, X_test):
        """
        Makes predictions using the Linear Regression model.
        
        This method uses the learned weights to predict target values for the test data:
        
            y_pred = w0 + w1X1 + ... + wnXn =  W * X_augmented
        
        Args:
            X_test (numpy.ndarray): The test data features.
        
        Returns:
            numpy.ndarray: The predicted target values.
        """
        X = np.hstack((np.ones(X_test.shape), X_test))
        y_pred = np.dot(X, self.weights)
        return y_pred