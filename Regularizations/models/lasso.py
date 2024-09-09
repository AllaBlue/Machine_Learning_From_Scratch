import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression

class LassoCoordinate():
    """
    A class that implements Lasso regression using the Coordinate Descent algorithm.

    Args:
        alpha (float): Regularization strength. Must be a positive float. Regularization improves
            the conditioning of the problem and reduces the variance of the estimates. Larger values 
            specify stronger regularization. Default is 1.

    Attributes:
        coef_ (numpy.ndarray): Coefficients of the features in the decision function, shape (n_features,).
        intercept_ (float): The independent term in the decision function.
        sc (StandardScaler): Scaler to standardize the input features.
    """

    def __init__(self, alpha: float = 1) -> None:
        """
        Initialize the LassoCoordinate class with the given alpha value.
        
        Args:
            alpha (float): Regularization strength.
        """
        self.alpha = alpha
        # Lasso Coordinate Descent will NOT work if data is not standardized
        self.sc = StandardScaler()


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, iterations: int) -> 'LassoCoordinate':
        """
        Fits the Lasso regression model using the training data and coordinate descent.

        The algorithm proceeds as follows:
        1. Calculate residuals, which include all terms (weights * features) EXCEPT the current term.
        2. Calculate the gradient for the current weight only (excluding the intercept).
        3. Update the current weight using the soft-thresholding operator with the gradient and alpha.
        4. Update the intercept with the mean of the residuals, which centers the residuals around 0.
        
        Args:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).
            iterations (int): Number of iterations for the coordinate descent algorithm.
        
        Returns:
            LassoCoordinate: Returns self.
        """
        # Standardize the training data
        X_train = self.sc.fit_transform(X_train)
        _, n_features = X_train.shape

        # Initialize coefficients and intercept
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Perform coordinate descent for the specified number of iterations
        for _ in range(iterations):
            self.update_current_weight(X_train=X_train, y_train=y_train)
            self.update_bias(X_train=X_train, y_train=y_train)
        
        return self

    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict target values using the Lasso regression model.

        Args:
            X_test (numpy.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        # Standardize the test data
        X_test = self.sc.transform(X_test)

        # Compute the predicted values using the learned weights and intercept
        y_pred = self.intercept_ + np.dot(X_test, self.coef_)

        return y_pred


    def update_current_weight(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Update the weight for each feature using the coordinate descent algorithm.

        Args:
            X_train (numpy.ndarray): Training data of shape (n_samples, n_features).
            y_train (numpy.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X_train.shape
        
        for feature in range(n_features):

            # Calculate the residual fixing all weights except the weight associated with the current feature
            # for example:
            # residual = y_train - w0 - w1x1 - w2x2 
            residual = y_train - (np.dot(X_train, self.coef_) + self.intercept_)
            # except feature
            # residual = y_train - w0 - w1x1 - w2x2 + w1x1 (in case we fix x1 feature)
            current_residual = residual + X_train[:, feature] * self.coef_[feature]
            
            # Calculate the gradient for the current weight
            z = np.dot(X_train[:, feature], current_residual) / n_samples
            pho = self.alpha

            # Update the current weight using the soft thresholding operator
            current_weight = self.soft_thresholding_operator(z=z, pho=pho)

            # Update the weight for the current feature
            self.coef_[feature] = current_weight


    def update_bias(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Update the bias (intercept) term using the mean of the residuals.

        Args:
            X_train (numpy.ndarray): Training data of shape (n_samples, n_features).
            y_train (numpy.ndarray): Target values of shape (n_samples,).
        """
        # Update the intercept by centering the residuals around zero
        self.intercept_ = np.mean(y_train - np.dot(X_train, self.coef_))


    def soft_thresholding_operator(self, z: float, pho: float) -> float:
        """
        Apply the soft thresholding operator to the given value.

        Args:
            z (float): The value to apply the soft thresholding operator to.
            pho (float): The regularization parameter (alpha).

        Returns:
            float: The result of the soft thresholding operation.
        """
        if z > pho:
            return z - pho
        elif np.abs(z) <= pho:
            return 0
        elif z < (-1) * pho:
            return z + pho
        

class LassoGradient():
    """
    A class that implements Lasso regression using Gradient Descent.

    Args:
        alpha (float): Regularization strength. Must be a positive float. Regularization improves
            the conditioning of the problem and reduces the variance of the estimates. Larger values 
            specify stronger regularization. Default is 1.
        learning_rate (float): Step size for each iteration in the gradient descent algorithm. Default is 0.01.

    Attributes:
        coef_ (numpy.ndarray): Coefficients of the features in the decision function, shape (n_features,).
        intercept_ (float): The independent term in the decision function.
        sc (StandardScaler): Scaler to standardize the input features.
    """

    def __init__(self, alpha: float = 1, learning_rate: float = 0.01) -> None:
        """
        Initialize the LassoGradient class with the given alpha and learning_rate.

        Args:
            alpha (float): Regularization strength.
            learning_rate (float): The step size for each iteration in the gradient descent algorithm.
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.sc = StandardScaler()


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, iterations: int) -> 'LassoGradient':
        """
        Fit the Lasso regression model using the training data and gradient descent.

        Args:
            X_train (numpy.ndarray): Training data of shape (n_samples, n_features).
            y_train (numpy.ndarray): Target values of shape (n_samples,).
            iterations (int): Number of iterations for the gradient descent algorithm.

        Returns:
            LassoGradient: Returns self.
        """
        _, n_features = X_train.shape

        # Standardize the training data
        X_train = self.sc.fit_transform(X_train)

        # Initialize coefficients and intercept
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Perform gradient descent for the specified number of iterations
        for _ in range(iterations):
            gradient_weights = self.get_gradient_weights(X_train=X_train, y_train=y_train)
            gradient_bias = self.get_gradient_bias(X_train=X_train, y_train=y_train)

            # Update the coefficients and intercept using the gradients
            self.coef_ = self.coef_ - self.learning_rate * gradient_weights
            self.intercept_ = self.intercept_ - self.learning_rate * gradient_bias
        
        return self


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict target values using the Lasso regression model.

        Args:
            X_test (numpy.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        # Standardize the test data
        X_test = self.sc.transform(X_test)
        # Compute the predicted values using the learned weights and intercept
        y_pred = self.intercept_ + np.dot(X_test, self.coef_)
        return y_pred


    def get_gradient_weights(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Compute the gradient for the weights using the training data.

        Args:
            X_train (numpy.ndarray): Training data of shape (n_samples, n_features).
            y_train (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            numpy.ndarray: The gradient of the weights, shape (n_features,).
        """
        n_samples, _ = X_train.shape

        # Calculate the residuals
        residual = y_train - (np.dot(X_train, self.coef_) + self.intercept_)

        # Calculate the gradient for the weights
        gradient_weights = ((-1)/n_samples) * np.dot(X_train.T, residual)

        # Adjust the gradient for Lasso regularization (+ or - alpha)
        gradient_weights = gradient_weights + self.alpha * np.sign(self.coef_)

        return gradient_weights
    

    def get_gradient_bias(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Compute the gradient for the bias (intercept) using the training data.

        Args:
            X_train (numpy.ndarray): Training data of shape (n_samples, n_features).
            y_train (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            float: The gradient of the bias.
        """
        n_samples, _ = X_train.shape

        # Calculate the residuals
        residual = y_train - (np.dot(X_train, self.coef_) + self.intercept_)

        # Calculate the gradient for the bias
        gradient_bias = ((-1)/n_samples) * sum(residual)

        return gradient_bias
    

# lasso = LassoCoordinate(alpha=0.6)
# lasso_gradient = LassoGradient(alpha=0.6, learning_rate=0.1)
# lasso_sklearn = Lasso(alpha=0.6)

# X = np.array([1,2,3,4,2,4,6,8,1,5,1,5]).reshape(3,4)
# y = np.array([2,5,6])

# sc = StandardScaler()
# X = sc.fit_transform(X)

# lasso.fit(X,y,1000)
# lasso_sklearn.fit(X,y)
# lasso_gradient.fit(X,y,10000)

# print("lasso.coef_")
# print(lasso.coef_)

# print("lasso_sklearn.coef_")
# print(lasso_sklearn.coef_)

# print("lasso_gradient.coef_")
# print(lasso_gradient.coef_)
