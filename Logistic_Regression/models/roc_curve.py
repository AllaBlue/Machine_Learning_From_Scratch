import numpy as np
import matplotlib.pyplot as plt
from Logistic_Regression.models.logistic_metrics import LogisticMetrics

class RocCurvePlot():
    """
    A class to compute and plot Receiver Operating Characteristic (ROC) curves.

    This class provides methods to compute the ROC curve, determine the optimal threshold using Youden's Index, 
    and plot the ROC curve along with the area under the curve (AUC).

    Attributes:
        x1 (float): X-coordinate for the start point of the ROC curve.
        y1 (float): Y-coordinate for the start point of the ROC curve.
        x3 (float): X-coordinate for the end point of the ROC curve.
        y3 (float): Y-coordinate for the end point of the ROC curve.
    """

    def __init__(self):
        """
        Initializes the RocCurvePlot class.
        """
        self.x1, self.y1 = (0, 0)
        self.x3, self.y3 = (1, 1)

    @staticmethod
    def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the ROC curve for binary classification.

        Args:
            y_true (np.ndarray): True binary labels of shape (n_samples,).
            y_score (np.ndarray): Target scores or probabilities of shape (n_samples,).

        Returns:
            tuple: A tuple containing:
                - np.ndarray: False Positive Rates for each threshold.
                - np.ndarray: True Positive Rates for each threshold.
                - np.ndarray: Thresholds used to compute the FPR and TPR.
        """
        unique_labels = np.sort(np.unique(y_true))
        y_true_binary = np.array([np.where(unique_labels == i)[0][0] for i in y_true])
        # true positive rates
        tprs = []
        # false positive rates
        fprs = []

        thresholds = np.linspace(0, 1, 100)
        thresholds = thresholds[::-1]
        for threshold in thresholds:
            y_pred = np.array([1 if score >= threshold else 0 for score in y_score])
            tpr = LogisticMetrics.recall_label(y_true=y_true_binary, y_pred=y_pred, label=1)
            fpr = 1 - LogisticMetrics.recall_label(y_true=y_true_binary, y_pred=y_pred, label=0)

            tprs.append(tpr)
            fprs.append(fpr)

        return np.array(fprs), np.array(tprs), thresholds
    
    @staticmethod
    def optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[int, float]:
        """
        Finds the optimal threshold using Youden's Index.

        Args:
            fpr (np.ndarray): False Positive Rates of shape (n_thresholds,).
            tpr (np.ndarray): True Positive Rates of shape (n_thresholds,).
            thresholds (np.ndarray): Thresholds used to compute the FPR and TPR.

        Returns:
            tuple: A tuple containing:
                - int: Index of the optimal threshold.
                - float: Optimal threshold value.
        """
        J = tpr - fpr
        optimal_index = np.argmax(J)
        optimal_threshold = thresholds[optimal_index]

        return optimal_index, optimal_threshold
    
    @staticmethod
    def plot(fpr: np.ndarray, tpr: np.ndarray, auc: float, fpr_threshold: float, tpr_threshold: float, 
             optimal_threshold: float, ax: plt.Axes, title: str) -> None:
        """
        Plots the ROC curve.

        Args:
            fpr (np.ndarray): False Positive Rates of shape (n_thresholds,).
            tpr (np.ndarray): True Positive Rates of shape (n_thresholds,).
            auc (float): Area Under the Curve.
            fpr_threshold (float): False Positive Rate at the optimal threshold.
            tpr_threshold (float): True Positive Rate at the optimal threshold.
            optimal_threshold (float): Optimal threshold value.
            ax (plt.Axes): The axes on which to plot.
            title (str): Title of the plot.
        """
        plt.suptitle(t="ROC Curve", color="lightcoral", fontsize=28)
        ax.set_title(label=title, color="lightcoral", fontsize=24, pad=15)
        
        ax.plot(fpr, tpr, color="lightcoral", lw=2, label=f"AUC = {np.round(auc, 2)}")
        ax.scatter(x=fpr_threshold, y=tpr_threshold, color="lavenderblush", s=200, 
                    edgecolors="lightcoral", zorder=2, marker="D",
                    label=f"optimal threshold = {np.round(optimal_threshold, 2)}")
        
        ax.set_xlabel(xlabel="False Positive Rate: Class 1", fontsize=16)
        ax.set_ylabel(ylabel="True Positive Rate: Class 1", fontsize=16)

        # ax.set_xticks(ax.get_xticks(), fontsize=14)
        # ax.set_yticks(ax.get_yticks(), fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        ax.legend(markerscale=0.75)

    @staticmethod
    def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Computes the Area Under the ROC Curve (AUC) using integral calculations.

        The area under the ROC curve (AUC) is calculated by summing the areas of trapezoids formed between consecutive points on the ROC curve. The integral method used here involves approximating the area under the curve by breaking it into trapezoids and summing their areas.

        Args:
            fpr (np.ndarray): False Positive Rates of shape (n_thresholds,).
            tpr (np.ndarray): True Positive Rates of shape (n_thresholds,).

        Returns:
            float: The computed AUC value.
        """

        auc = 0

        for index, _ in enumerate(fpr):
            if index != (len(fpr) - 1):
                x1 = fpr[index]
                x2 = fpr[index + 1]
                y1 = tpr[index]
                y2 = tpr[index + 1]

                if x2==x1:
                    continue

                # Compute the slope of the line segment (the tangent)
                tangent = (y2 - y1) / (x2 - x1)

                # Area of the trapezoid is computed using the integral of the line segment
                # y = m(x - x1) + y1, where m is the slope (tangent)
                # Integral of y from x1 to x2 gives the area under the line segment

                # Integral calculation:
                # area = integral of (m(x - x1) + y1) dx from x1 to x2
                #       = [m(x^2/2 - x1*x) + y1*x] from x1 to x2
                #       = x2(m(x2/2 - x1) + y1) - x1(m(x1/2 - x1) + y1)
                
                area = x2 * (tangent * (x2/2 - x1) + y1) - x1 * (tangent * (x1/2 - x1) + y1)
                # Add the calculated area to the total AUC
                auc += area

        return auc    
    


