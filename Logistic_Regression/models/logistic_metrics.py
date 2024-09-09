import numpy as np
import pandas as pd

class LogisticMetrics():
    """
    A collection of static methods for calculating common metrics used in logistic regression 
    evaluation, including accuracy, precision, recall, F1 score, and confusion matrix.

    These methods are designed to work with both binary and multiclass classification problems.
    """

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the accuracy of the predictions.

        Accuracy is the proportion of correct predictions over the total number of predictions.

        Args:
            y_true (np.ndarray): Array of true labels.
            y_pred (np.ndarray): Array of predicted labels.

        Returns:
            float: Accuracy score.
        """
        return np.mean(y_true == y_pred)
    

    @staticmethod
    def true_positive(y_true: np.ndarray, y_pred: np.ndarray, label) -> int:
        """
        Computes the number of true positives for a given label.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            int: Count of true positives.
        """
        return np.sum((y_true == y_pred) & (y_true == label))
    

    @staticmethod
    def true_negative(y_true: np.ndarray, y_pred: np.ndarray, label) -> int:
        """
        Computes the number of true negatives for a given label.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            int: Count of true negatives.
        """
        return np.sum((y_true != label) & (y_pred != label))
    

    @staticmethod
    def false_positive(y_true: np.ndarray, y_pred: np.ndarray, label) -> int:
        """
        Computes the number of false positives for a given label.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            int: Count of false positives.
        """
        return np.sum((y_true != label) & (y_pred == label))
    

    @staticmethod
    def false_negative(y_true: np.ndarray, y_pred: np.ndarray, label) -> int:
        """
        Computes the number of false negatives for a given label.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            int: Count of false negatives.
        """
        return np.sum((y_true == label) & (y_pred != label))
    

    @staticmethod
    def recall_label(y_true: np.ndarray, y_pred: np.ndarray, label) -> float:
        """
        Calculates recall for a specific label.

        Recall is the proportion of actual positives that were identified correctly.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            float: Recall score.
        """
        true_positive = LogisticMetrics.true_positive(y_true=y_true, y_pred=y_pred, label=label)
        false_negative = LogisticMetrics.false_negative(y_true=y_true, y_pred=y_pred, label=label)
        # total actual positive
        total_positive_labels = true_positive + false_negative

        return true_positive / total_positive_labels
    

    @staticmethod
    def precision_label(y_true: np.ndarray, y_pred: np.ndarray, label) -> float:
        """
        Calculates precision for a specific label.

        Precision is the proportion of positive predictions that were actually correct.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            float: Precision score.
        """
        true_positive = LogisticMetrics.true_positive(y_true=y_true, y_pred=y_pred, label=label)
        false_positive = LogisticMetrics.false_positive(y_true=y_true, y_pred=y_pred, label=label)
        # total predicted positive
        total_positive_guesses = true_positive + false_positive

        return true_positive / total_positive_guesses
    

    @staticmethod
    def f1_score_label(y_true: np.ndarray, y_pred: np.ndarray, label) -> float:
        """
        Calculates the F1 score for a specific label.

        The F1 score is the harmonic mean of precision and recall.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.
            label: The label of interest.

        Returns:
            float: F1 score.
        """
        recall = LogisticMetrics.recall_label(y_true=y_true, y_pred=y_pred, label=label)
        precission = LogisticMetrics.precision_label(y_true=y_true, y_pred=y_pred, label=label)
        
        return (2 * recall * precission) / (recall + precission)
    

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the macro-average recall for all labels.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.

        Returns:
            float: Macro-average recall score.
        """
        labels = np.unique(y_true)
        total_recall = 0

        for label in labels:
            total_recall += LogisticMetrics.recall_label(y_true=y_true, y_pred=y_pred, label=label)
        
        recall = total_recall / len(labels)

        return recall
    

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the macro-average precision for all labels.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.

        Returns:
            float: Macro-average precision score.
        """
        labels = np.unique(y_true)
        total_precission = 0

        for label in labels:
            total_precission += LogisticMetrics.precision_label(y_true=y_true, y_pred=y_pred, label=label)
        
        precission = total_precission / len(labels)

        return precission
    

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the macro-average F1 score for all labels.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.

        Returns:
            float: Macro-average F1 score.
        """
        labels = np.unique(y_true)
        total_f1_score = 0

        for label in labels:
            total_f1_score += LogisticMetrics.f1_score_label(y_true=y_true, y_pred=y_pred, label=label)
        
        f1_score = total_f1_score / len(labels)

        return f1_score
    

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Generates a confusion matrix for multiclass classification.

        The confusion matrix shows the counts of true positives, false positives, 
        true negatives, and false negatives for each label.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.

        Returns:
            numpy.ndarray: A 2D confusion matrix with shape (n_labels, n_labels).
        """
        labels = np.unique(y_true)

        matrix = np.zeros(shape=(len(labels), len(labels)))
 
        for i, _ in enumerate(labels):
            for j, label_column in enumerate(labels):
                # on main diagonal all true positive of the label corresponding to the current column
                if i == j:
                    matrix[i][j] = LogisticMetrics.true_positive(y_true=y_true, y_pred=y_pred, label=label_column)
                # otherwise it is false positive of the label correspnsing to the current column
                else:
                    matrix[i][j] = LogisticMetrics.false_positive(y_true=y_true, y_pred=y_pred, label=label_column)

        
        return matrix
    

    @staticmethod
    def classification_metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Generates a comprehensive classification metrics report.

        The report includes precision, recall, F1 score, and support for each label, 
        along with overall accuracy, macro-averaged precision, recall, and F1 score.

        Args:
            y_true (numpy.ndarray): Array of true labels.
            y_pred (numpy.ndarray): Array of predicted labels.

        Returns:
            pandas.DataFrame: A DataFrame containing classification metrics for each label and overall metrics.
        """
        unique_labels = np.sort(np.unique(y_true))

        data = []

        for label in unique_labels:

            label_data = {
                "precision": np.round(LogisticMetrics.precision_label(y_true=y_true, y_pred=y_pred, label=label), 2),
                "recall": np.round(LogisticMetrics.recall_label(y_true=y_true, y_pred=y_pred, label=label), 2),
                "f1_score": np.round(LogisticMetrics.f1_score_label(y_true=y_true, y_pred=y_pred, label=label), 2),
                "support": np.sum(y_true == label).astype(int)
            }

            label_data_df = pd.DataFrame(data=label_data, index=[label])
            data.append(label_data_df)

        df = pd.concat(data)

        accuracy_data = {
            "precision": "",
            "recall": "",
            "f1_score": np.round(LogisticMetrics.accuracy(y_true=y_true, y_pred=y_pred), 2),
            "support": int(y_true.shape[0])
        }

        macro_average_data = {
            "precision": np.round(LogisticMetrics.precision(y_true=y_true, y_pred=y_pred), 2),
            "recall": np.round(LogisticMetrics.recall(y_true=y_true, y_pred=y_pred), 2),
            "f1_score": np.round(LogisticMetrics.f1_score(y_true=y_true, y_pred=y_pred), 2),
            "support": int(y_true.shape[0])
        }

        df.loc["accuracy"] = pd.Series(accuracy_data)
        df.loc["macro_average"] = pd.Series(macro_average_data)
        
        return df