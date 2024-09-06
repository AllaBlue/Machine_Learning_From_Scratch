from Linear_Regression.models.standard_scaler import StandardScaler as SC
from Polynomial_Regression.models.polynomial_features import PolynomialFeatures as PF
from Linear_Regression.models.linear_regression import LinearRegression as LR
from Linear_Regression.models.regression_metrics import RegressionMetrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class ModelsUpdate():
    """
    A class to compare custom and sklearn polynomial regression models by evaluating them 
    on standardized data across multiple polynomial degrees and storing the results.

    Attributes:
        maes_custom (list): Mean absolute errors for custom models.
        mses_custom (list): Mean squared errors for custom models.
        rmses_custom (list): Root mean squared errors for custom models.
        r2_custom (list): R^2 scores for custom models.
        preds_custom (list): Predictions from custom models.

        maes_sklearn (list): Mean absolute errors for sklearn models.
        mses_sklearn (list): Mean squared errors for sklearn models.
        rmses_sklearn (list): Root mean squared errors for sklearn models.
        r2_sklearn (list): R^2 scores for sklearn models.
        preds_sklearn (list): Predictions from sklearn models.

        models (list): Information about custom and sklearn models, including evaluation metrics.
    """

    def __init__(self):
        """
        Initializes the ModelsUpdate class with empty lists for storing evaluation metrics 
        and predictions for both custom and sklearn models.
        """

        # Custom model metrics
        self.maes_custom = []
        self.mses_custom = []
        self.rmses_custom = []
        self.r2_custom = []
        self.preds_custom = []

        # Sklearn model metrics
        self.maes_sklearn = []
        self.mses_sklearn = []
        self.rmses_sklearn = []
        self.r2_sklearn = []
        self.preds_sklearn = []

        # Stores model information for comparison
        self.models = []


    def standardize(self, X_train, X_test):
        """
        Standardizes the training and testing data using both custom and sklearn scalers.

        Args:
            X_train (np.ndarray): Training data features of shape (n_samples, n_features).
            X_test (np.ndarray): Testing data features of shape (n_samples, n_features).

        Returns:
            tuple: Standardized training and testing data for both custom and sklearn scalers, 
                   in the order (X_train_custom, X_test_custom, X_train_sklearn, X_test_sklearn).
        """

        # Custom standardization
        sc_custom = SC()
        X_train_custom = sc_custom.fit_transform(X_train)
        X_test_custom = sc_custom.transform(X_test)

        # Sklearn standardization
        sc_sklearn = StandardScaler()
        X_train_sklearn = sc_sklearn.fit_transform(X_train)
        X_test_sklearn = sc_sklearn.transform(X_test)

        return X_train_custom, X_test_custom, X_train_sklearn, X_test_sklearn
    
    def get_metrics(self, X_train, X_test, y_train, y_test, degrees=11):
        """
        Computes and stores regression metrics (MAE, MSE, RMSE, R^2) for polynomial 
        regression models of various degrees using both custom and sklearn implementations.

        Args:
            X_train (np.ndarray): Training data features of shape (n_samples, n_features).
            X_test (np.ndarray): Testing data features of shape (n_samples, n_features).
            y_train (np.ndarray): Training data target values of shape (n_samples,).
            y_test (np.ndarray): Testing data target values of shape (n_samples,).
            degrees (int): Maximum degree of the polynomial features to be considered. Default is 11.

        Returns:
            None
        """

        # Standardize the data
        X_train_custom, X_test_custom, X_train_sklearn, X_test_sklearn = self.standardize(X_train, X_test)

        # Clear previous metrics
        self.maes_custom = []
        self.mses_custom = []
        self.rmses_custom = []
        self.r2_custom = []
        self.preds_custom = []

        self.maes_sklearn = []
        self.mses_sklearn = []
        self.rmses_sklearn = []
        self.r2_sklearn = []
        self.preds_sklearn = []

        # Loop through each degree to compute metrics
        for degree in range(1, degrees):

            # Custom polynomial feature transformation
            pf_custom = PF(degree=degree)
            X_train_pf_custom = pf_custom.fit_transform(X_train_custom)
            X_test_pf_custom = pf_custom.transform(X_test_custom)

            # Sklearn polynomial feature transformation
            pf_sklearn = PolynomialFeatures(degree=degree)
            X_train_pf_sklearn = pf_sklearn.fit_transform(X_train_sklearn)
            X_test_pf_sklearn = pf_sklearn.transform(X_test_sklearn)

            # Fit custom linear regression model and predict
            self.lr_custom = LR()
            self.lr_custom.fit(X_train_pf_custom, y_train)
            pred_custom = self.lr_custom.predict(X_test_pf_custom)
            self.preds_custom.append(pred_custom)

            # Fit sklearn linear regression model and predict
            self.lr_sklern = LinearRegression()
            self.lr_sklern.fit(X_train_pf_sklearn, y_train)
            pred_sklearn = self.lr_sklern.predict(X_test_pf_sklearn)
            self.preds_sklearn.append(pred_sklearn)


            # Calculate and store custom model metrics
            self.maes_custom.append(RegressionMetrics.MAE(y_true=y_test, y_pred=pred_custom))
            self.mses_custom.append(RegressionMetrics.MSE(y_true=y_test, y_pred=pred_custom))
            self.rmses_custom.append(RegressionMetrics.RMSE(y_true=y_test, y_pred=pred_custom))
            self.r2_custom.append(RegressionMetrics.R2(y_true=y_test, y_pred=pred_custom))

            # Calculate and store sklearn model metrics
            self.maes_sklearn.append(mean_absolute_error(y_true=y_test, y_pred=pred_sklearn))
            self.mses_sklearn.append(mean_squared_error(y_true=y_test, y_pred=pred_sklearn))
            self.rmses_sklearn.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=pred_sklearn)))
            self.r2_sklearn.append(r2_score(y_true=y_test, y_pred=pred_sklearn))
        
    
    def update_models(self):
        """
        Updates the `models` attribute with information about the custom and sklearn models, 
        including metrics for each model.

        Returns:
            None
        """

        self.models = [
            {
                "model": self.lr_custom,
                "maes": self.maes_custom,
                "mses": self.mses_custom,
                "rmses": self.rmses_custom,
                "r2": self.r2_custom,
                "title": "Custom",
                "y_preds": self.preds_custom,
                "color": "aquamarine",
                "edgecolor": "aquamarine",
                "scattercolor": "darkslategrey"
                
            },
            {
                "model": self.lr_sklern,
                "maes": self.maes_sklearn,
                "mses": self.mses_sklearn,
                "rmses": self.rmses_sklearn,
                "r2": self.r2_sklearn,
                "title": "Sklearn",
                "y_preds": self.preds_sklearn,
                "color": "khaki",
                "edgecolor": "khaki",
                "scattercolor": "sienna"
            }
        ]

        # Define metrics to evaluate
        metrics = [
            {
                "name": "maes",
                "title": "Mean Absolute Error",
            },
            {
                "name": "mses",
                "title": "Mean Squared Error"
            },
            {
                "name": "rmses",
                "title": "Root Mean Squared Error",
            },
            {
                "name": "r2",
                "title": "R^2 Score"
            },
        ]

        # Compute and store the best metrics for each model
        for model in self.models:
            for metric in metrics:
                # Determine whether to find the minimum or maximum value
                measure = "max" if metric["name"] == "r2" else "min"
                func1 = min if measure == "min" else max
                func2 = np.argmin if measure == "min" else np.argmax
                model[f"{metric["name"]}_{measure}"] = func1(model[metric["name"]])
                model[f"{metric["name"]}_arg{measure}"] = func2(model[metric["name"]])