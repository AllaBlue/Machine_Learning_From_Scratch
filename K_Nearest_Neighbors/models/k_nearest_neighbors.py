import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

import numpy as np

class KNearestNeighbors:
    """
    K-Nearest Neighbors (KNN) classifier implemented from scratch.

    This class provides methods to train a KNN model, predict labels, 
    and predict label probabilities for a given dataset. It supports 
    both categorical and numerical data as target labels.
    
    Attributes:
        n_neighbors (int): The number of neighbors to use for classification.
    """

    def __init__(self, n_neighbors: int = 5):
        """
        Initializes the KNN classifier with a specified number of neighbors.

        Args:
            n_neighbors (int): Number of neighbors to use for classification. Default is 5.
        """
        self.n_neighbors = n_neighbors
        self.predict_flag = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the KNN model to the training data.

        Stores the training data and encodes the target labels.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        self.X_train = X
        self.y_train = self.__encode_y(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the input data based on the nearest neighbors.

        Args:
            X (np.ndarray): Input data to classify.

        Returns:
            np.ndarray: Predicted labels for each input sample.
        """
        n_samples = X.shape[0]
        self.y_pred = [0] * n_samples
        self.y_pred_proba = [0] * n_samples
        
        for index, element in enumerate(X):
            label, probabilities = self.predict_sample(element)
            self.y_pred[index] = label
            self.y_pred_proba[index] = probabilities
            
        self.y_pred_proba = np.array(self.y_pred_proba)
        
        self.predict_flag = True

        return self.__decode_y(self.y_pred)
    
    def predict_proba(self, X: np.ndarray = None) -> np.ndarray:
        """
        Predicts the probability distributions over classes for input samples.

        Args:
            X (np.ndarray, optional): Input data to classify. If omitted, 
            the method assumes that `predict` has already been called.

        Returns:
            np.ndarray: Probability distribution for each input sample across all classes.
        """
        if self.predict_flag:
            return self.y_pred_proba
        else:
            self.predict(X)
            return self.y_pred_proba
    
    def predict_sample(self, X: np.ndarray) -> tuple:
        """
        Predicts the label and probability distribution for a single input sample.

        Args:
            X (np.ndarray): Single input sample.

        Returns:
            tuple: Predicted label and probability distribution over classes.
        """
        distances = np.linalg.norm(self.X_train - X, axis=1)
        
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        top_k = self.y_train[nearest_indices]

        label = np.bincount(top_k, minlength=len(self.unique_labels)).argmax()
        probabilities = np.bincount(top_k, minlength=len(self.unique_labels)) / len(top_k)

        return label, probabilities
    
    def __encode_y(self, y: np.ndarray) -> np.ndarray:
        """
        Encodes categorical labels to integer values.

        Args:
            y (np.ndarray): Array of labels.

        Returns:
            np.ndarray: Encoded integer values of labels.
        """
        self.unique_labels = np.sort(np.unique(y))
        encoded_y = np.array([np.where(self.unique_labels == element)[0][0] for element in y])

        return encoded_y

    def __decode_y(self, y: np.ndarray) -> np.ndarray:
        """
        Decodes integer values back to original categorical labels.

        Args:
            y (np.ndarray): Array of encoded integer labels.

        Returns:
            np.ndarray: Decoded original labels.
        """
        decoded_y = np.array([self.unique_labels[element] for element in y])
        return decoded_y



###TEST###
# knn = KNearestNeighbors(n_neighbors=3)
# X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,5,6,3,7,2,8,1,1,1,8]).reshape(8,3)
# y = np.array(["one", "one", "one", "two", "two", "one", "one", "one"])
# test = np.array([2,2,2, 14,15,16, 7,3,8, 11,10,11]).reshape(4,3)
# knn.fit(X, y)
# prediction = knn.predict(test)
# probabilities = knn.predict_proba(test)
# print(prediction)
# print(probabilities)


#Sklearn
# knn_sklearn = KNeighborsClassifier(n_neighbors=3)
# knn_sklearn.fit(X,y)
# preds = knn_sklearn.predict(test)
# print(preds)









