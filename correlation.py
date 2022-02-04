import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_correlation(data: pd.DataFrame):
    plt.figure(figsize=(25, 25))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap=plt.cm.Reds)
    plt.show()
