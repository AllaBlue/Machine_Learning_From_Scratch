import numpy as np

class GradienDescent():
    """
    A custom implementation of the Gradient Descent algorithm for linear regression.

    The loss function used for optimization is Mean Squared Error (MSE), defined as:
        MSE = (1/n) * sum((y - XW)^2)

    The gradient of the loss function with respect to the weights is given by:
        Gradient = (-2/n) * sum(X(y - XW))

    Attributes:
        weights (np.ndarray): The weights of the linear regression model, initialized as None.
    """

    def __init__(self) -> None:
        """
        Initializes the GradientDescent class.
        """

        self.weights = None

    def get_gradient(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of the loss function with respect to the weights.

        Args:
            X_train (np.ndarray): The input features for training, with shape (n_samples, n_features).
            y_train (np.ndarray): The target values for training, with shape (n_samples,).
            weights (np.ndarray): The current weights of the model, with shape (n_features + 1,).

        Returns:
            np.ndarray: The computed gradient, with shape (n_features + 1,).
        """

        n_samples, _ = X_train.shape
        sum = 0

        # Compute the gradient sum for all samples
        for sample in range(n_samples):
            sum = sum + np.dot(X_train[sample, :], y_train[sample] - np.dot(X_train[sample, :], weights)) 
        
        gradient = ((-2) / n_samples) * sum

        return gradient
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, iterations: int) -> 'GradienDescent':
        """
        Fits the linear regression model using gradient descent.

        Args:
            X_train (np.ndarray): The input features for training, with shape (n_samples, n_features).
            y_train (np.ndarray): The target values for training, with shape (n_samples,).
            learning_rate (float): The step size for gradient descent updates.
            iterations (int): The number of iterations to perform gradient descent.

        Returns:
            None
        """

        n_samples, n_features = X_train.shape
        X = np.hstack((np.ones(shape=(n_samples, 1)), X_train))

        # Initialize weights if not already set
        if self.weights == None:
            self.weights = [0] * (n_features + 1)

        # Perform gradient descent for a specified number of iterations
        for _ in range(iterations):
            gradient = self.get_gradient(X_train=X, y_train=y_train, weights=self.weights)
            self.weights = self.weights - learning_rate * gradient
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given test data.

        Args:
            X_test (np.ndarray): The input features for testing, with shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted target values, with shape (n_samples,).
        """
        
        n_samples, _ = X_test.shape
        X = np.hstack((np.ones(shape=(n_samples, 1)), X_test))
        y_pred = np.dot(X, self.weights)
        return y_pred

        

