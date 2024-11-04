import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Apply dark background style for plots
style.use("dark_background")

class Plot():
    """
    A class for creating various types of plots with customized styling.
    """

    @staticmethod
    def set_plot_params(figsize: tuple = (16, 8)) -> None:
        """
        Set global plot parameters for customizing the appearance of plots.
        
        Args:
            figsize (tuple): A tuple specifying the figure size (width, height). Default is (16, 8).
        
        Returns:
            None
        """
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.titlesize"] = 28
        plt.rcParams["axes.titlesize"] = 18
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["figure.autolayout"] = True


    @staticmethod
    def histplot(data: pd.DataFrame, rows: int, columns: int, color: str, hue: str = None) -> None:
        """
        Create a histogram plot for each feature in the DataFrame.
        
        Args:
            data (pd.DataFrame): DataFrame containing the features to plot.
            rows (int): Number of rows in the subplot grid.
            columns (int): Number of columns in the subplot grid.
            color (str): Color for the histograms and titles.
            hue (str, optional): Feature name for grouping data by color. Default is None.
        
        Returns:
            None
        """
        plt.suptitle(t="Distributions", color=color)

        for index, feature in enumerate(data.columns):
            plt.subplot(rows, columns, index + 1)
            plt.title(label=feature, color=color, pad=15)

            # check if the feature is numeric
            if is_numeric_dtype(data[feature]):
                if hue is not None:
                    sns.histplot(data=data, x=feature, kde=True, palette=f"dark:{color}", hue=hue)
                else:
                    sns.histplot(data=data, x=feature, kde=True, color=color)
            else:
                if hue is not None:
                    sns.countplot(data=data, x=feature, palette=f"dark:{color}", hue=hue)
                else:
                    sns.countplot(data=data, x=feature, color=color)


    @staticmethod
    def feature_vs_target(data: pd.DataFrame, target: str, rows: int, columns: int, color: str, scatter_color: str) -> None:
        """
        Create scatter plots of each feature versus the target variable.
        
        Args:
            data (pd.DataFrame): DataFrame containing the features and target.
            target (str): Name of the target variable.
            rows (int): Number of rows in the subplot grid.
            columns (int): Number of columns in the subplot grid.
            color (str): Color for the subplot titles and axis labels.
            scatter_color (str): Color for the scatter plot points.
        
        Returns:
            None
        """
        plt.suptitle(t="Feature vs Target", color=color)
        
        for index, feature in enumerate(data.columns):
            plt.subplot(rows, columns, index + 1)
            plt.scatter(x=data[feature], y=data[target], s=100, edgecolors=color, color=scatter_color)
            plt.title(label=feature, color=color, pad=15)
            plt.xlabel(xlabel=feature)
            plt.ylabel(ylabel=target)
    

    @staticmethod
    def train_vs_test_accuracy(train_test_params: list, x: np.ndarray, x_label: str, suptitle: str) -> None:
        """
        Plot accuracy of training vs. testing models.
        
        Args:
            train_test_params (list of dict): List containing dictionaries with keys:
                - 'title': Title of the subplot.
                - 'color': Color for the line plot.
                - 'scatter_color': Color for the scatter plot points.
                - 'y': Accuracy values to plot.
            x (np.ndarray): X-values for plotting.
            x_label (str): Label for the x-axis.
            suptitle (str): Title for the entire figure.
        
        Returns:
            None
        """
        plt.figure(figsize=(16,8))
        plt.suptitle(t=suptitle)

        for index, params in enumerate(train_test_params):
            plt.subplot(1, 2, index + 1)
            plt.title(label=f"{params["title"]} Accuracy", color=params["color"], pad=15)
            plt.scatter(x=x, y=params["y"], linewidths=2,
                        color=params["scatter_color"], edgecolors=params["color"], s=200, zorder=3)
            # Scatter maximum point
            plt.scatter(x=x[np.argmax(params["y"])], y=np.max(params["y"]), linewidths=2,
                        s=400, color="pink", edgecolors="purple", zorder=4, label="Maximum Accuracy")

            plt.plot(x, params["y"], 
                     color=params["color"], linewidth=3)
            plt.xlabel(xlabel=x_label)
            plt.ylabel(ylabel="Accuracy")
            plt.legend(markerscale=0.75)
    
    @staticmethod
    def heatmap_features_target(df, target, cmap):
        correlation_matrix = df.corr()[[target]]
        correlation_matrix["absolute_corr"] = abs(correlation_matrix)

        heatmap = sns.heatmap(correlation_matrix.drop(index=target)
                            .sort_values(by="absolute_corr", ascending=False)
                            .drop(columns=["absolute_corr"]), 
                            annot=True, annot_kws={"fontsize": 15, "weight": "bold"}, cmap=cmap)
        heatmap.set_title(f"Features Correlating with {target}", fontsize=28)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=15)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=15, rotation=0)
        plt.tight_layout()
