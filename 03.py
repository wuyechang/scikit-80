from sklearn import datasets

# pylint: disable=no-member

iris = datasets.load_iris()

data = iris.data

print(data.shape)
print(iris.feature_names)

target = iris.target

print(target.shape)
print(iris.target_names)