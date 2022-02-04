import numpy as np
import pandas as pd
import math
import seaborn as sns
import csv

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# Initialize train and test data sets
# Returns: X_train, y_train, X_test, y_test
def init_data(data: pd.DataFrame, reduce=False):
    if reduce:
        data = reduce_data(data)

    return split_data(data)


# Splits the input data set into train and test subsets
def split_data(data: pd.DataFrame, train_size=0.8):
    x = data.drop(['Class'], axis=1)
    y = data['Class']

    return train_test_split(x, y, train_size=train_size, stratify=y, random_state=0)


# Reduce the amount Class0 tuples. Reduce the skew towards Class0
def reduce_data(data: pd.DataFrame):
    # Get subsets of tuples for each class
    class_zero = data[data['Class'] == 0]
    class_one = data[data['Class'] == 1]

    # Configurable value for how many Class0 tuples to keep when reducing
    reduce_ratio = 1.5

    # Calculate how many Class0 tuples to sample and sample them
    class_one_count = len(class_one)
    class_zero_keep = math.floor(class_one_count * reduce_ratio)
    class_zero = class_zero.sample(class_zero_keep, replace=False)

    return pd.concat([class_zero, class_one])
