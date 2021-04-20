import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train = pd.read_csv("titanic_train.csv")

# basic data analysis

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.show()

sns.countplot(x="Survived", data=train, hue="Pclass", palette="RdBu_r")
plt.show()

sns.distplot(train["Age"].dropna(), kde=False, bins=30)
plt.show()

sns.countplot(x="SibSp", data=train)
plt.show()

train["Fare"].hist(bins=40, figsize=(10, 4))
plt.show()


# data cleaning
def impute_age(cols):
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


train["Age"] = train[["Age", "Pclass"]].apply(impute_age, axis=1)

train.drop("Cabin", axis=1, inplace=True)
train.dropna(inplace=True)

sex = pd.get_dummies(train["Sex"], drop_first=True)

embark = pd.get_dummies(train["Embarked"], drop_first=True)

train = pd.concat([train, sex, embark], axis=1)

train.drop(["Sex", "Embarked", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

# machine learning shizwoz
y = train["Survived"]
X = train.drop("Survived", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

lm = LogisticRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

print(classification_report(y_test, predictions))
