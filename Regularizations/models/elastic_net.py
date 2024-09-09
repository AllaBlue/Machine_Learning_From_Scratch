import numpy as np
from sklearn.preprocessing import StandardScaler

class ElasticNetCoordinate():
    """
    Elastic Net Coordinate Descent algorithm for regression.

    The objective function is given by:
        L = (1/2n) * sum((y - (XW + b))^2) + alpha * l1_ratio * |W| + 1/2 * alpha * (1-l1_ratio) |W|^2

    Attributes:
        alpha (float): Regularization strength.
        l1_ratio (float): Ratio of L1 to L2 regularization.
        sc (StandardScaler): Scaler for standardizing features.
        coef_ (numpy.ndarray): Coefficients of the fitted model.
        intercept_ (float): Intercept (bias) term of the fitted model.
    """

    def __init__(self, alpha: float = 1, l1_ratio: float = 0.5) -> None:
        """
        Initializes the ElasticNetCoordinate class.

        Args:
            alpha (float): Regularization strength.
            l1_ratio (float): Ratio of L1 to L2 regularization.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.sc = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, iterations: int) -> 'ElasticNetCoordinate':
        """
        Fits the Elastic Net model using coordinate descent.

        Args:
            X_train (np.ndarray): Training data features of shape (n_samples, n_features).
            y_train (np.ndarray): Training data target values of shape (n_samples,).
            iterations (int): Number of iterations for the coordinate descent algorithm.

        Returns:
            ElasticNetCoordinate: Returns self
        """
        _, n_features = X_train.shape
        
        # Standardize features
        X_train = self.sc.fit_transform(X_train)

        # Initialize coefficients and intercept
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Coordinate descent iterations
        for _ in range(iterations):
            self.update_current_weight(X_train=X_train, y_train=y_train)
            self.update_bias(X_train=X_train, y_train=y_train)
        
        return self
    

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts using the fitted Elastic Net model.

        Args:
            X_test (np.ndarray): Test data features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        X_test = self.sc.transform(X_test)
        y_pred = self.intercept_ + np.dot(X_test, self.coef_)

        return y_pred


    def update_current_weight(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Updates the weights using the coordinate descent algorithm.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target values.

        Returns:
            None
        """
        n_samples, n_features = X_train.shape

        for feature in range(n_features):
            
            # Calculate residual
            residual = y_train - (np.dot(X_train, self.coef_) + self.intercept_)
            # Exclude the contribution of the current weight
            current_residual = residual + np.dot(X_train[:, feature], self.coef_[feature])

            z = (1/n_samples) * np.dot(X_train[:, feature], current_residual)
            pho = self.alpha * self.l1_ratio

            scalar = 1/(((1/n_samples) * np.dot(X_train[:, feature].T, X_train[:, feature])) + self.alpha * (1 - self.l1_ratio))

            # Update current weight using soft-thresholding
            self.coef_[feature] = scalar * self.soft_thresholding_operator(z=z, pho=pho)
    

    def update_bias(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Updates the bias (intercept) term of the model.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target values.

        Returns:
            None
        """
        self.intercept_ = np.mean(y_train - np.dot(X_train, self.coef_))
    

    def soft_thresholding_operator(self, z: float, pho: float) -> float:
        """
        Soft-thresholding operator for L1 regularization.

        Args:
            z (float): Input value.
            pho (float): Threshold parameter.

        Returns:
            float: Thresholded value.
        """
        if z > pho:
            return z - pho
        elif np.abs(z) <= pho:
            return 0
        elif z < (-1) * pho:
            return z + pho


class ElasticNetGradient():
    """
    Elastic Net Gradient Descent algorithm for regression.

    The objective function is given by:
        L = (1/2n) * sum((y - (XW + b))^2) + alpha * l1_ratio * |W| + 1/2 * alpha * (1-l1_ratio) |W|^2

    Attributes:
        alpha (float): Regularization strength.
        l1_ratio (float): Ratio of L1 to L2 regularization.
        learning_rate (float): Learning rate for gradient descent.
        sc (StandardScaler): Scaler for standardizing features.
        coef_ (numpy.ndarray): Coefficients of the fitted model.
        intercept_ (float): Intercept (bias) term of the fitted model.
    """

    def __init__(self, alpha: float = 1, l1_ratio: float = 0.5, learning_rate: float = 0.01) -> None:
        """
        Initializes the ElasticNetGradient class.

        Args:
            alpha (float): Regularization strength.
            l1_ratio (float): Ratio of L1 to L2 regularization.
            learning_rate (float): Learning rate for gradient descent.
        """
        self.sc = StandardScaler()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, iterations: int) -> 'ElasticNetGradient':
        """
        Fits the Elastic Net model using gradient descent.

        Args:
            X_train (np.ndarray): Training data features of shape (n_samples, n_features).
            y_train (np.ndarray): Training data target values of shape (n_samples,).
            iterations (int): Number of iterations for gradient descent.

        Returns:
            ElasticNetGradient: Returns self
        """
        X_train = self.sc.fit_transform(X_train)

        _, n_features = X_train.shape

        # Initialize coefficients and intercept
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Gradient descent iterations
        for _ in range(iterations):
            gradient_weights = self.get_gradient_weights(X_train=X_train, y_train=y_train)
            gradient_bias = self.get_gradient_bias(X_train=X_train, y_train=y_train)

            # Update weights and bias
            self.coef_ = self.coef_ - self.learning_rate * gradient_weights
            self.intercept_ = self.intercept_ - self.learning_rate * gradient_bias
        
        return self
    

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts using the fitted Elastic Net model.

        Args:
            X_test (np.ndarray): Test data features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        X_test = self.sc.transform(X_test)
        y_pred = self.intercept_ + np.dot(X_test, self.coef_)

        return y_pred


    def get_gradient_weights(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of the weights for gradient descent.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target values.

        Returns:
            np.ndarray: Gradient of the weights.
        """
        n_samples, _ = X_train.shape

        residual = y_train - (np.dot(X_train, self.coef_) + self.intercept_)
        gradient_weights = (((-1/n_samples) * np.dot(X_train.T, residual))
                             + np.sign(self.coef_) * self.alpha * self.l1_ratio 
                             + self.alpha * (1 - self.l1_ratio) * self.coef_)
        
        return gradient_weights
    

    def get_gradient_bias(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Calculates the gradient of the bias (intercept) for gradient descent.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target values.

        Returns:
            float: Gradient of the bias.
        """
        n_samples, _ = X_train.shape

        residual = y_train - (np.dot(X_train, self.coef_) + self.intercept_)
        gradient_bias = (-1/n_samples) * np.sum(residual)
        
        return gradient_bias
    

# coordinate = ElasticNetCoordinate(alpha=0)
# gradient = ElasticNetGradient(alpha=0, learning_rate=0.1)
# en_sklearn = ElasticNet(alpha=0)

# X = np.array([1,2,3,4,2,4,6,8,1,5,1,5]).reshape(3,4)
# y = np.array([2,5,6])

# sc = StandardScaler()
# X = sc.fit_transform(X)

# coordinate.fit(X,y,1000)
# gradient.fit(X,y,10000)
# en_sklearn.fit(X,y)

# print("coordinate.coef_")
# print(coordinate.coef_)

# print("en_sklearn.coef_")
# print(en_sklearn.coef_)

# print("gradient.coef_")
# print(gradient.coef_)