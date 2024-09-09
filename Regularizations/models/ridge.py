import numpy as np


class RidgeRegularization():
    """
    A class to perform Ridge Regression, a linear regression model with L2 regularization 
    to prevent overfitting by penalizing large coefficients.

    Args:
        alpha (float, optional): Regularization strength. Must be a positive float. 
            Regularization improves the conditioning of the problem and reduces the variance 
            of the estimates. Larger values specify stronger regularization. Default is 1.

    Attributes:
        coef_ (numpy.ndarray): Coefficients of the features in the decision function, shape (n_features,).
        intercept_ (float): The independent term (bias) in the decision function.
        weights (numpy.ndarray): Learned weights for the model, including the bias term, shape (n_features + 1,).
    """
     
    def __init__(self, alpha: float = 1) -> None:
        """
        Initializes the RidgeRegularization class with the given regularization strength.

        Args:
            alpha (float, optional): Regularization strength. Default is 1.
        """

        self.alpha = alpha

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'RidgeRegularization':
        """
        Fits the Ridge regression model using the training data.

        Args:
            X_train (numpy.ndarray): Training data of shape (n_samples, n_features).
            y_train (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            RidgeRegularization: Returns self.
        """

        n_samples, _ = X_train.shape

        # Add bias term (intercept) to the input features by appending a column of ones
        X = np.hstack((np.ones(shape=(n_samples, 1)), X_train))
        
        # Identity matrix for regularization (excluding the bias term from regularization)
        I = np.identity(X.shape[1])
        I[0,0] = 0 # No Regularization for bias term

        # Calculate weights using the closed-form solution with Ridge regularization
        self.weights = np.dot(np.linalg.inv(self.alpha * I + np.dot(np.transpose(X), X)),
                               np.dot(np.transpose(X), y_train))
        
        # Extract the coefficients and intercept from the weights
        self.coef_ = self.weights[1:] # Coefficients for the features
        self.intercept_ = self.weights[0] # Intercept (bias) term

        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the Ridge regression model.

        Args:
            X_test (numpy.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """

        n_samples, _ = X_test.shape

        # Add bias term (intercept) to the input features by appending a column of ones
        X = np.hstack((np.ones(shape=(n_samples, 1)), X_test))
        
        # Calculate the predicted values using the learned weights
        y_pred = np.dot(X, self.weights)

        return y_pred
