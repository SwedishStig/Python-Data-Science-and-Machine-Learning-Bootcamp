import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("loan_data.csv")

plt.figure(figsize=(10, 6))
df[df["credit.policy"] == 1]["fico"].hist(bins=35, color="blue", label="Credit Policy = 1", alpha=.6)
df[df["credit.policy"] == 0]["fico"].hist(bins=35, color="red", label="Credit Policy = 0", alpha=.6)
plt.xlabel("FICO")
plt.show()

plt.figure(figsize=(10, 6))
df[df["not.fully.paid"] == 1]["fico"].hist(bins=35, color="blue", label="Not Fully Paid = 1", alpha=.6)
df[df["not.fully.paid"] == 0]["fico"].hist(bins=35, color="red", label="Not Fully Paid = 0", alpha=.6)
plt.xlabel("FICO")
plt.show()

plt.figure(figsize=(11, 7))
sns.countplot(data=df, x="purpose", hue="not.fully.paid", palette="Set1")
plt.show()

sns.jointplot(data=df, x="fico", y="int.rate", color="purple")
plt.show()

plt.figure(figsize=(11, 7))
sns.lmplot(data=df, y="int.rate", x="fico", hue="credit.policy", col="not.fully.paid", palette="Set1")
plt.show()

cat_feats = ["purpose"]
final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)

X = df.drop("not.fully.paid", axis=1)
y = df["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
