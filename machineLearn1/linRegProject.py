import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

customers = pd.read_csv("Ecommerce Customers")

sns.jointplot(data=customers, x="Time on Website", y="Yearly Amount Spent")
plt.show()

sns.jointplot(data=customers, x="Time on App", y="Yearly Amount Spent")
plt.show()

sns.jointplot(data=customers, x="Time on App", y="Length of Membership", kind="hex")
plt.show()

sns.pairplot(customers)
plt.show()

sns.lmplot(data=customers, x="Length of Membership", y="Yearly Amount Spent")
plt.show()

y = customers["Yearly Amount Spent"]
X = customers[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("Y test (true values)")
plt.ylabel("Predicted values")
plt.show()

from sklearn import metrics

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(metrics.explained_variance_score(y_test, predictions))

sns.distplot((y_test-predictions), bins=50)
plt.show()

cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coef"])

# App probably
