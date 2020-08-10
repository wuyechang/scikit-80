import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# pylint: disable=no-member

iris = datasets.load_iris()
data = iris.data
target = iris.target

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.scatter(data[:,0], data[:,1], c=target)

plt.subplot(122)

plt.scatter(data[:,2], data[:,3], c=target)

plt.show()