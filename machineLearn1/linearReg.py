import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("USA_Housing.csv")

# sns.pairplot(df)
# plt.show()
#
# sns.distplot(df["Price"])
# plt.show()
#
# sns.heatmap(df.corr())
# plt.show()

# print(df.columns)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coef"])

predictions = lm.predict(X_test)
sns.distplot((y_test - predictions))
plt.show()

from sklearn import metrics

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

