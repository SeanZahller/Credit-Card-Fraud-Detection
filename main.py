from modeling.knn import test_knn_classifier
from preprocess.correlation import get_correlation
import pandas as pd

# Load creditcard data
df = pd.read_csv("./data/creditcard.csv")

# Show pearson correlation matrix
get_correlation(df)

# KNN classification
knn_results = test_knn_classifier(df)
print(knn_results)
