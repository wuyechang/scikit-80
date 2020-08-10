from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=no-member

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

data_1 = iris_df['sepal length (cm)'].hist(bins=30)

# for class_number in np.unique(iris.target):
#     plt.figure(1)
#     iris_df['sepal length (cm)'].iloc[np.where(iris.target == class_number)[0]].hist(bins=30)