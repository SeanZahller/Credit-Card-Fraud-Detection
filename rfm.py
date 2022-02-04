import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score


from preprocess.load_data import init_data


def rfm():
    # Read in data
    data = pd.read_csv("creditcard.csv")

    #Split training and test sets, reduce size of data to make less skewed
    xTrain, xTest, yTrain, yTest = init_data(data, reduce=True)

    #Random Forest Model
    rfc = RandomForestClassifier()

    #Training the model to predict with the training sets
    rfc.fit(xTrain, yTrain)
    yPred = rfc.predict(xTest)

    print("Random Forest Model Classifier")

    acc = accuracy_score(yTest, yPred)
    print("Accuracy: {}".format(acc))

    prec = precision_score(yTest, yPred)
    print("Precision: {}".format(prec))

    rec = recall_score(yTest, yPred)
    print("Recall: {}".format(rec))



