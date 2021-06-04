import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_info = pd.read_csv("lending_club_info.csv", index_col="LoanStatNew")


def feat_info(col_name):
    print(data_info.lic[col_name]["Description"])


df = pd.read_csv("lending_club_loan_two.csv")

sns.countplot(x="loan_status", data=df)
plt.show()

plt.figure(figsize=(12, 4))
sns.distplot(df["loan_amnt"], kde=False, bins=40)
plt.show()

plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.ylim(10, 0)
plt.show()

sns.scatterplot(x="installment", y="loan_amnt", data=df)
plt.show()

sns.boxplot(x="loan_status", y="loan_amnt", data=df)
plt.show()

print(df.groupby("loan_status")["loan_amnt"].describe())

sns.countplot(x="grade", data=df, hue="loan_status")
plt.show()

plt.figure(figsize=(12, 4))
sub_order = sorted(df["sub_grade"].unique())
sns.countplot(x="sub_grade", data=df, order = sub_order, palette="coolwarm", hue="loan_status")
plt.show()

fg = df[(df["grade"] == "G") | (df["grade"] == "F")]
plt.figure(figsize=(12, 4))
sub_order = sorted(fg["sub_grade"].unique())
sns.countplot(x="sub_grade", data=fg, order = sub_order, palette="coolwarm")
plt.show()

df["loan_repaid"] = df["loan_status"].map({"Fully Paid" : 1, "Charged Off" : 0})

df.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")
plt.show()

print(df.isnull().sum())
print(100 * df.isnull().sum() / len(df))

df = df.drop("emp_title", axis=1)

plt.figure(figsize=(12, 4))
sns.countplot(x="emp_length", data=df, hue="loan_status")
plt.show()

emp_co = df[df["loan_status"] == "Charged Off"].groupby("emp_length").count()["loan_status"]
emp_fp = df[df["loan_status"] == "Fully Paid"].groupby("emp_length").count()["loan_status"]
emp_len = emp_co/(emp_co + emp_fp)
print(emp_len)
emp_len.plot(kind="bar")
plt.show()

df = df.drop("emp_length", axis=1)

df = df.drop("title", axis=1)

print(df.corr()["mort_acc"].sort_values())

total_acc_avg = df.groupby("total_acc").mean()["mort_acc"]


def fill_mort(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df["mort_acc"] = df.apply(lambda x: fill_mort(x["total_acc"], x["mort_acc"]), axis=1)

df = df.dropna()

df["term"] = df["term"].apply(lambda t: int(t[:3]))

df = df.drop("grade", axis=1)

dummies = pd.get_dummies(df["sub_grade"], drop_first=True)
df = pd.concat([df.drop("sub_grade", axis=1), dummies], axis=1)

dummies = pd.get_dummies(df[["verification_status", "application_type", "initial_list_status", "purpose"]], drop_first=True)
df = pd.concat([df.drop(["verification_status", "application_type", "initial_list_status", "purpose"], axis=1), dummies], axis=1)

df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")

dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
df = pd.concat([df.drop("home_ownership", axis=1), dummies], axis=1)

df["zip_code"] = df["address"].apply(lambda t: t[-5:])

dummies = pd.get_dummies(df["zip_code"], drop_first=True)
df = pd.concat([df.drop("zip_code", axis=1), dummies], axis=1)

df = df.drop("address", axis=1)

df = df.drop("issue_d", axis=1)

df["earliest_cr_line"] = df["earliest_cr_line"].apply(lambda t: int(t[-4:]))

from sklearn.model_selection import train_test_split

df = df.drop("loan_status", axis=1)

X = df.drop("loan_repaid", axis=1).values
y = df["loan_repaid"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(78, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(39, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(19, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))

losses = pd.DataFrame(model.history.history)

losses.plot()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

import random
random.seed(101)
random_ind = random.randint(0, len(df))

new_cust = df.drop("loan_repaid", axis=1).iloc[random_ind]

new_cust = scaler.transform(new_cust.values.reshape(1, 78))

print(model.predict_classes(new_cust))
