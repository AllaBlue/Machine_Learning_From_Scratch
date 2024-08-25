import numpy as np


class RidgeRegularization():
    """
    A class to perform Ridge Regression, which is a type of linear regression that includes
    a regularization term to prevent overfitting.

    Parameters
    ----------
    alpha : float, optional (default=1)
        Regularization strength. Must be a positive float. Regularization improves
        the conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        The coefficients of the features in the decision function.
    
    intercept_ : float
        The independent term in the decision function.
    
    weights : ndarray of shape (n_features + 1,)
        The learned weights for the model, including the bias term.
    """
     
    def __init__(self, alpha=1):
        """
        Initialize the RidgeRegularization class with the given alpha value.
        
        Parameters
        ----------
        alpha : float, optional (default=1)
            Regularization strength.
        """

        self.alpha = alpha

    def fit(self, X_train, y_train):
        """
        Fit the Ridge regression model using the training data.
        
        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Training data.
        
        y_train : ndarray of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
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
    
    def predict(self, X_test):
        """
        Predict target values using the Ridge regression model.
        
        Parameters
        ----------
        X_test : ndarray of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """

        n_samples, _ = X_test.shape

        # Add bias term (intercept) to the input features by appending a column of ones
        X = np.hstack((np.ones(shape=(n_samples, 1)), X_test))
        
        # Calculate the predicted values using the learned weights
        y_pred = np.dot(X, self.weights)

        return y_pred
