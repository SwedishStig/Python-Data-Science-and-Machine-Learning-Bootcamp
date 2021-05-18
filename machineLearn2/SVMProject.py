import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

iris = sns.load_dataset('iris')     # dataset not found?

sns.pairplot(iris, hue="species", palette="Dark2")
plt.show()

setosa = iris[iris["species"] == "setosa"]
sns.kdeplot(setosa["sepal_width"], setosa["sepal_length"], cmap="plasma", shade=True, shade_lowest=False)
plt.show()

X = iris.drop("species", axis=1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

param_grid = {"C": [.1, 1, 10, 100], "gamma": [1, .1, .01, .001]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)

grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
