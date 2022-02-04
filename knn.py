from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from preprocess.load_data import init_data
from math import sqrt
import pandas as pd


def test_knn_classifier(data: pd.DataFrame):
    # TODO: implement iterative model evaluation from 1:n neighbrors
    # for n in range(1, 5):
    #     accuracy, rmse = knn(data, n)
    #     print(n, accuracy, rmse)

    accuracy, rmse = knn(data, 1)

    print(accuracy, rmse)


def knn(data: pd.DataFrame, n: int):
    x_train, x_test, y_train, y_test = init_data(data, reduce=True)

    knn_model = KNeighborsRegressor(n_neighbors=n)
    knn_model.fit(x_train, y_train)

    # Calculate the RMSE
    train_pred = knn_model.predict(x_train)
    mse = mean_squared_error(y_train, train_pred)
    rmse = sqrt(mse)

    # Get prediction for test data
    test_pred = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, test_pred)

    return accuracy, rmse

