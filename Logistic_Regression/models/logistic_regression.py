import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler

class LogisticRegression():
    """
    Logistic Regression using gradient descent.

    The objective function (log-loss) for binary logistic regression is:
        L(W, b) = -(1/n) * Σ [y * log(p) + (1 - y) * log(1 - p)]
    where:
        - p = sigmoid(XW + b) is the predicted probability,
        - X is the feature matrix,
        - W are the weights, and
        - b is the bias term.

    The gradients for the weights and bias are:
        Gradient_W = (1/n) * Σ X.T * (p - y)
        Gradient_b = (1/n) * Σ (p - y)

    Attributes:
        coef_ (np.ndarray): Learned coefficients for the features.
        intercept_ (float): Learned bias term.
        learning_rate (float): Step size for gradient descent.
        iterations (int): Number of iterations for gradient descent.
        loss_ (list): List of loss values at each iteration.
    """

    def __init__(self, learning_rate: float, iterations: int) -> None:
        """
        Initializes the LogisticRegression class with a learning rate and number of iterations.

        Args:
            learning_rate (float): The step size for gradient descent updates.
            iterations (int): Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the logistic regression model using gradient descent.

        Args:
            X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            None
        """
        self.get_labels(y=y)

        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        self.loss_ = []

        for _ in range(self.iterations):
            # calculate loss
            y_pred_prob = self.sigmoid(z=np.dot(X, self.coef_) + self.intercept_)
            self.loss_.append(((-1)/n_samples) * (np.dot(self.y_binary, np.log(y_pred_prob))
                                                   + np.dot((1 - self.y_binary), np.log(1 - y_pred_prob))))
            
            # calculate gradients
            gradient_weights = self.get_gradient_weights(X=X, y=self.y_binary)
            gradient_bias = self.get_gradient_bias(X=X, y=self.y_binary)

            # update weights and bias
            self.coef_ = self.coef_ - self.learning_rate * gradient_weights
            self.intercept_ = self.intercept_ - self.learning_rate * gradient_bias
    

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each class for the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities for each class, shape (n_samples, 2).
        """
        probabilities_of_positive = self.sigmoid(z=np.dot(X, self.coef_) + self.intercept_)
        probabilities = np.zeros(shape=(probabilities_of_positive.shape[0], 2))

        for index, probability in enumerate(probabilities_of_positive):
            # firstly show probabilities of 0, then 1
            probabilities[index] = [1 - probability, probability]

        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        probabilities = self.predict_proba(X=X)
        # Convert probabilities to binary class predictions
        y_pred = np.argmax(probabilities, axis=1)

        return y_pred
    

    def get_gradient_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the weights for logistic regression.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Binary target labels of shape (n_samples,).

        Returns:
            np.ndarray: Gradient of the weights, shape (n_features,).
        """
        n_samples, _ = X.shape
        y_pred = self.sigmoid(z=np.dot(X, self.coef_) + self.intercept_)
        
        return (1/n_samples) * np.dot(X.T, y_pred - y)


    def get_gradient_bias(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the gradient of the bias for logistic regression.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Binary target labels of shape (n_samples,).

        Returns:
            float: Gradient of the bias.
        """
        n_samples, _ = X.shape 
        y_pred = self.sigmoid(z=np.dot(X, self.coef_) + self.intercept_)

        return (1/n_samples) * np.sum(y_pred - y)
    

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function to the input.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid transformation of the input.
        """
        return 1/(1 + np.exp(-z))
    
    
    def get_labels(self, y: np.ndarray) -> None:
        """
        Converts the target labels to binary values for logistic regression.

        Args:
            y (np.ndarray): Target labels.

        Returns:
            None
        """
        self.unique_labels = np.sort(np.unique(y))
        # convert labels into two classes: 0 and 1 -> get index of true class from unique labels
        self.y_binary = np.array([np.where(self.unique_labels == i)[0][0] for i in y])

    


class MultinomialRegression():
    """
    Multinomial Logistic Regression using gradient descent.

    The objective function (log-loss) for multinomial logistic regression is:
        L(W, b) = -(1/n) * Σ Σ y_k * log(p_k)
    where:
        - p_k = softmax(XW + b)_k is the predicted probability for class k,
        - y_k is a one-hot encoded vector for class k.

    The gradients for the weights and bias are:
        Gradient_W = (1/n) * Σ X.T * (p_k - y_k)
        Gradient_b = (1/n) * Σ (p_k - y_k)

    Attributes:
        coef_ (np.ndarray): Learned coefficients for each class.
        intercept_ (np.ndarray): Learned bias term for each class.
        learning_rate (float): Step size for gradient descent.
        iterations (int): Number of iterations for gradient descent.
        loss_ (list): List of loss values at each iteration.
    """

    def __init__(self, learning_rate: float, iterations: int) -> None:
        """
        Initializes the MultinomialRegression class with a learning rate and number of iterations.

        Args:
            learning_rate (float): The step size for gradient descent.
            iterations (int): Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialRegression':
        """
        Fits the multinomial logistic regression model using gradient descent.

        Args:
            X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            MultinomialRegression: Returns self
        """

        n_samples, _ = X.shape

        self.get_labels(y=y)
        self.init_weights(X=X)
        self.init_bias()
        one_hot_y = self.one_hot_encoder(y=y)

        self.loss_ = []

        for _ in range(self.iterations):
            # calculate gradients
            gradient_weights = self.get_gradient_weights(X=X, y=one_hot_y)
            gradient_bias = self.get_gradient_bias(X=X, y=one_hot_y)

            # calculate loss
            y_pred_proba = self.predict_proba(X=X)
            self.loss_.append( ((-1)/n_samples) * np.sum(one_hot_y * np.log(y_pred_proba)))

            # update weights and bias
            self.coef_ = self.coef_ - self.learning_rate * gradient_weights
            self.intercept_ = self.intercept_ - self.learning_rate * gradient_bias
        
        return self

    def one_hot_encoder(self, y: np.ndarray) -> np.ndarray:
        """
        Encodes the target labels as one-hot vectors.

        Args:
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            np.ndarray: One-hot encoded labels of shape (n_samples, n_classes).
        """
        one_hot_y = np.zeros(shape=(y.shape[0], len(self.unique_labels)))

        for index, label in enumerate(y):
            label_index = np.where(self.unique_labels == label)[0][0]
            one_hot_y_row = np.zeros(shape=(1, len(self.unique_labels)))
            one_hot_y_row[0][label_index] = 1
            one_hot_y[index] = one_hot_y_row
        
        return one_hot_y
    

    def init_weights(self, X: np.ndarray) -> None:
        """
        Initializes the weights for the multinomial regression model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            None
        """
        _, n_features = X.shape

        self.coef_ = np.zeros(shape=(len(self.unique_labels), n_features))
    
    def init_bias(self) -> None:
        """
        Initializes the bias for the multinomial regression model.

        Returns:
            None
        """
        self.intercept_ = np.zeros(shape=(len(self.unique_labels)))


    def get_gradient_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the weights for multinomial logistic regression.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): One-hot encoded target labels of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Gradient of the weights, shape (n_classes, n_features).
        """
        n_samples, _ = X.shape
        y_pred_proba = self.predict_proba(X=X)

        gradient_weights = (1/n_samples) * np.dot(X.T, (y_pred_proba - y))

        return gradient_weights.T
    
    def get_gradient_bias(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the bias for multinomial logistic regression.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): One-hot encoded target labels of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Gradient of the bias, shape (n_classes,).
        """
        n_samples, _ = X.shape

        y_pred_proba = self.predict_proba(X=X)
        gradient_bias = (1/n_samples) * np.sum(y_pred_proba - y)

        return gradient_bias.T
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        probabilities = self.predict_proba(X=X)
        y_pred = np.zeros(shape=probabilities.shape[0])

        for index, _ in enumerate(y_pred):
            y_pred[index] = self.unique_labels[np.argmax(probabilities[index])]

        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each class for the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities for each class, shape (n_samples, n_classes).
        """
        n_samples, _ = X.shape
        y_pred = np.zeros(shape=(len(self.unique_labels), n_samples))

        for index, label in enumerate(self.one_hot_labels):
            y_pred[index] = self.predict_label_proba(X=X, one_hot_label=label)
        
        return y_pred.T


    def predict_label_proba(self, X: np.ndarray, one_hot_label: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of a specific class label.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            one_hot_label (np.ndarray): One-hot encoded label.

        Returns:
            np.ndarray: Probabilities of the specified class.
        """
        # get logit for this particular label (or class)
        logit_label = self.logit_label(X=X, one_hot_label=one_hot_label)

        normalization_sum = 0
        
        for label in self.one_hot_labels:
            logit = self.logit_label(X=X, one_hot_label=label)
            normalization_sum += np.exp(logit)
        
        y_pred_label = np.exp(logit_label) / normalization_sum

        return y_pred_label


    def logit_label(self, X: np.ndarray, one_hot_label: np.ndarray) -> np.ndarray:
        """
        Computes the logit (pre-softmax) for a specific class label.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            one_hot_label (np.ndarray): One-hot encoded label.

        Returns:
            np.ndarray: Logit values for the specified class label.
        """
        label_index = np.where(one_hot_label == 1)[0][0]

        logit_label = np.dot(X, self.coef_[label_index]) + self.intercept_[label_index]

        return logit_label
    
    def get_labels(self, y: np.ndarray) -> None:
        """
        Converts the target labels to numerical values for multinomial regression.

        Args:
            y (np.ndarray): Target labels.

        Returns:
            None
        """
        self.unique_labels = np.sort(np.unique(y))
        self.one_hot_labels = self.one_hot_encoder(y=self.unique_labels)
        


# -------------------------------------------------------------------------------------
#                            Testing of Logistic Regression
# -------------------------------------------------------------------------------------


# logit = LogisticRegression(learning_rate=0.01, iterations=1)
# lr = LR()

# X = np.array([1,2,3,4,2,4,6,8,1,5,1,5,9, 12,3,4]).reshape(4,4)
# y = np.array([0, 0, 1, 1])

# sc = StandardScaler()
# X = sc.fit_transform(X)

# logit.fit(X,y)
# lr.fit(X,y)

# print("logit.coef_")
# print(logit.coef_)

# print("lr.coef_")
# print(lr.coef_)

# print("logit.intercept_")
# print(logit.intercept_)

# print("lr.intercept_")
# print(lr.intercept_)

# print("logit.predict_proba(X)")
# print(logit.predict_proba(X))

# print("lr.predict_proba(X)")
# print(lr.predict_proba(X))

# print("logit.loss_")
# print(logit.loss_)


# -------------------------------------------------------------------------------------
#                            Testing of Multinomial Regression
# -------------------------------------------------------------------------------------


# X = np.arange(start=0, stop=40, step=1).reshape(10, 4)
# y = np.hstack((np.ones(4) * 1, np.ones(3) * 2, np.ones(3) * 3))
# mr = MultinomialRegression(learning_rate=0.01, iterations=1)
# lr = LR()

# lr.fit(X, y)
# mr.fit(X,y)

# print("mr.predict_proba(X)")
# print(mr.predict_proba(X))

# print("lr.predict_proba(X)")
# print(lr.predict_proba(X))

# print("mr.predict(X)")
# print(mr.predict(X))

# print("lr.predict(X)")
# print(lr.predict(X))


