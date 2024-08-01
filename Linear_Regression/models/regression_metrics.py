import numpy as np
import pandas as pd

class RegressionMetrics():
    """
    A class to calculate various regression evaluation metrics.
    
    Methods:
        MSE(y_true, y_pred): Calculate Mean Square Error.
        RMSE(y_true, y_pred): Calculate Root Mean Square Error.
        MAE(y_true, y_pred): Calculate Mean Absolute Error.
        R2(y_true, y_pred): Calculate R2 score.
        print_metrics(y_true, y_pred): Print all regression metrics.
        metrics_dataframe(y_true, y_pred): Return all regression metrics as a DataFrame.
    """

    @staticmethod
    def MSE(y_true, y_pred):
        """
        Calculate Mean Square Error (MSE).
        
        MSE measures the average of the squares of the errors, i.e., the average squared difference 
        between the actual and predicted values.

        Formula:
            MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true (numpy.ndarray): The actual target values.
            y_pred (numpy.ndarray): The predicted target values.
        
        Returns:
            float: The Mean Square Error.
        """
        return np.mean(np.power((y_true - y_pred), 2))
    

    @staticmethod
    def RMSE(y_true, y_pred):
        """
        Calculate Root Mean Square Error (RMSE).
        
        RMSE is the square root of the average of the squares of the errors. It provides an 
        indication of the magnitude of the errors.

        Formula:
            RMSE = √((1/n) * Σ(y_true - y_pred)²)
        
        Args:
            y_true (numpy.ndarray): The actual target values.
            y_pred (numpy.ndarray): The predicted target values.
        
        Returns:
            float: The Root Mean Square Error.
        """
        return np.sqrt(RegressionMetrics.MSE(y_true, y_pred))
    

    @staticmethod
    def MAE(y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE measures the average magnitude of the errors in a set of predictions, without considering 
        their direction. It's the average over the test sample of the absolute differences between 
        prediction and actual observation where all individual differences have equal weight.

        Formula:
            MAE = (1/n) * Σ|y_true - y_pred|
        
        Args:
            y_true (numpy.ndarray): The actual target values.
            y_pred (numpy.ndarray): The predicted target values.
        
        Returns:
            float: The Mean Absolute Error.
        """
        return np.mean(np.abs(y_true - y_pred))
    

    @staticmethod
    def R2(y_true, y_pred):
        """
        Calculate the R2 score (Coefficient of Determination).
        
        R2 score provides an indication of goodness of fit and therefore a measure of how well 
        unseen samples are likely to be predicted by the model. The best possible score is 1.0, 
        and it can be negative (because the model can be arbitrarily worse).

        Formula:
            R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)
        
        Args:
            y_true (numpy.ndarray): The actual target values.
            y_pred (numpy.ndarray): The predicted target values.
        
        Returns:
            float: The R2 score.
        """
        return 1 - np.sum(np.power((y_true - y_pred), 2)) / np.sum(np.power((y_true - np.mean(y_true)), 2))
    

    @staticmethod
    def print_metrics(y_true, y_pred):
        """
        Print all regression metrics: MSE, RMSE, MAE, and R2 score.
        
        Args:
            y_true (numpy.ndarray): The actual target values.
            y_pred (numpy.ndarray): The predicted target values.
        """
        print(f"MSE:     {RegressionMetrics.MSE(y_true, y_pred).round(2)}")
        print(f"RMSE:    {RegressionMetrics.RMSE(y_true, y_pred).round(2)}")
        print(f"MAE:     {RegressionMetrics.MAE(y_true, y_pred).round(2)}")
        print(f"R2:      {RegressionMetrics.R2(y_true, y_pred).round(2)}")
    

    @staticmethod
    def metrics_dataframe(y_true, y_pred):
        """
        Return all regression metrics as a pandas DataFrame.
        
        Args:
            y_true (numpy.ndarray): The actual target values.
            y_pred (numpy.ndarray): The predicted target values.
        
        Returns:
            pandas.DataFrame: A DataFrame containing MSE, RMSE, MAE, and R2 score.
        """
        data = {"MSE": RegressionMetrics.MSE(y_true, y_pred).round(2),
                "RMSE": RegressionMetrics.RMSE(y_true, y_pred).round(2),
                "MAE": RegressionMetrics.MAE(y_true, y_pred).round(2),
                "R2": RegressionMetrics.R2(y_true, y_pred).round(2)
                }
        return pd.DataFrame(data=data, index=[0])

