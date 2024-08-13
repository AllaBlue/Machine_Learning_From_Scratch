import numpy as np

class PolynomialFeatures():
    
    def __init__(self, degree):
        self.degree = degree
        self.combinations = []


    def fit(self, X):
        _, n_features = X.shape

        # Generate combinations of features for each degree
        for d in range(1, self.degree + 1):
            for combination in self.combinations_with_replacement_(range(n_features), d):
                self.combinations.append(combination)
    

    def transform(self, X):

        transformed_features = []

        for combination in self.combinations:
            feature = np.prod(X[:, combination], axis=1)
            transformed_features.append(feature)
        
        return np.column_stack(transformed_features)
    
    def fit_transform(self, X):
        
        self.fit(X)
        return self.transform(X)
    
    def combinations_with_replacement_(self, iterable, r):
        pool = tuple(iterable)
        n = len(pool)

        indices = [0] * r

        yield(tuple(pool[i] for i in indices))

        while True:
            for i in reversed(range(r)):
                if indices[i] != n-1:
                    break
            else:
                return
        
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