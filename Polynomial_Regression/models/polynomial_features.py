import numpy as np

class PolynomialFeatures():
    """
    Generate polynomial and interaction features.

    This class generates a new feature matrix consisting of all polynomial combinations 
    of the features with degree less than or equal to the specified degree. 
    For example, if an input sample is two-dimensional and of the form [a, b], 
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    degree : int
        The degree of polynomial features to generate.

    Attributes
    ----------
    degree : int
        The degree of polynomial features to generate.
    combinations : list
        A list to store generated combinations of feature indices.

    Methods
    -------
    fit(X)
        Generates polynomial feature combinations for the input array X.
    transform(X)
        Transforms the input array X by generating the polynomial features based on the combinations.
    fit_transform(X)
        Combines the fit and transform methods: generates polynomial features and returns the transformed array.
    combinations_with_replacement_(iterable, r)
        Helper function that generates combinations with replacement of the input iterable.
    """
    
    def __init__(self, degree):
        """
        Initializes the PolynomialFeatures class with a specified degree.

        Parameters
        ----------
        degree : int
            The degree of polynomial features to generate.
        """

        self.degree = degree
        self.combinations = []


    def fit(self, X):
        """
        Generates combinations of feature indices for the input array X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data array where n_samples is the number of samples and 
            n_features is the number of features.

        Returns
        -------
        None
        """

        _, n_features = X.shape

        # Generate combinations of features for each degree
        for d in range(1, self.degree + 1):
            for combination in self.combinations_with_replacement_(range(n_features), d):
                self.combinations.append(combination)
    

    def transform(self, X):
        """
        Transforms the input array X by generating polynomial features 
        based on the combinations generated in the fit method.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data array to be transformed.

        Returns
        -------
        numpy array of shape (n_samples, n_polynomial_features)
            The transformed array including the polynomial features.
        """

        transformed_features = []

        for combination in self.combinations:
            feature = np.prod(X[:, combination], axis=1)
            transformed_features.append(feature)
        
        return np.column_stack(transformed_features)
    
    def fit_transform(self, X):
        """
        Combines the fit and transform methods to generate polynomial features 
        and return the transformed array.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data array to be fitted and transformed.

        Returns
        -------
        numpy array of shape (n_samples, n_polynomial_features)
            The transformed array including the polynomial features.
        """
        
        self.fit(X)
        return self.transform(X)
    
    def combinations_with_replacement_(self, iterable, r):
        """
        Helper function that generates combinations with replacement 
        of the input iterable.

        Parameters
        ----------
        iterable : iterable
            The input iterable from which to generate combinations.
        r : int
            The number of elements in each combination.

        Yields
        ------
        tuple
            A tuple representing a combination with replacement.
        """

        pool = tuple(iterable)
        n = len(pool)

        # Initialize indices array
        indices = [0] * r

        yield(tuple(pool[i] for i in indices))

        while True:
            # Find the rightmost index that is not at its maximum value
            for i in reversed(range(r)):
                if indices[i] != n-1:
                    break
            else:
                return # All indices are at their maximum, stop iteration
            
            # Increment the current index and reset all following indices
            indices[i:] = [indices[i] + 1] * (r - i)

            yield(tuple(pool[i] for i in indices))
    
    

#Example
#combination = (0,0)
# X = np.array([[1,2], 
#               [3,4], 
#               [5,6]])
# print(X[:, combination])

# arr = [1,2,3,4]
# combinations(arr, 3)