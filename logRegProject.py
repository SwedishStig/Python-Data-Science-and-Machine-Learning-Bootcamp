import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ad_data = pd.read_csv("advertising.csv")

ad_data["Age"].plot.hist(bins=30)
plt.show()

sns.jointplot(data=ad_data, x="Age", y="Area Income")
plt.show()

sns.jointplot(data=ad_data, x="Age", y="Daily Time Spent on Site", kind="kde", color="red")
plt.show()

sns.jointplot(data=ad_data, x="Daily Time Spent on Site", y="Daily Internet Usage", color="green")
plt.show()

sns.pairplot(ad_data, hue="Clicked on Ad")
plt.show()

X = ad_data[["Daily Time Spent on Site", "Area Income", "Daily Internet Usage", "Male"]]
y = ad_data["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

lm = LogisticRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

print(classification_report(y_test, predictions))
