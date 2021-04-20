import numpy as np
import pandas as pd

import seaborn as sea
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv('911.csv')

print(df.info())

print(df.head())

print(df.info())

print(df["zip"].value_counts().head(5))

print(df["twp"].value_counts().head(5))

print(df['title'].nunique())

df["Reason"] = df['title'].apply(lambda re: re.split(':')[0])
print(df["Reason"].value_counts().head(1))

sea.countplot(x=df["Reason"], data=df, palette="viridis")
plt.show()

print(type(df["timeStamp"].iloc[0]))

df["timeStamp"] = pd.to_datetime(df["timeStamp"])
print(type(df["timeStamp"].iloc[0]))

df["Hour"] = df["timeStamp"].apply(lambda ti: ti.hour)
df["Month"] = df["timeStamp"].apply(lambda ti: ti.month)
df["Day of Week"] = df["timeStamp"].apply(lambda ti: ti.dayofweek)
# print(df["Hour"].iloc[0])

dmap = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["Day of Week"] = df["Day of Week"].map(dmap)

sea.countplot(x=df["Day of Week"], hue=df["Reason"], data=df, palette="viridis")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.show()

sea.countplot(x=df["Month"], hue=df["Reason"], data=df, palette="viridis")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.show()

# Some months are missing

byMonth = df.groupby("Month").count()
byMonth["lat"].plot()
plt.show()

sea.lmplot(x="Month", y="twp", data=byMonth.reset_index())
plt.show()

df["Date"] = df["timeStamp"].apply(lambda ti: ti.date())

df.groupby("Date").count()["lat"].plot()
plt.tight_layout()
plt.show()

df[df["Reason"] == "Traffic"].groupby("Date").count()["lat"].plot()
plt.title("Traffic")
plt.tight_layout()
plt.show()
df[df["Reason"] == "Fire"].groupby("Date").count()["lat"].plot()
plt.title("Fire")
plt.tight_layout()
plt.show()
df[df["Reason"] == "EMS"].groupby("Date").count()["lat"].plot()
plt.title("EMS")
plt.tight_layout()
plt.show()

dh = df.groupby(by=["Day of Week", "Hour"]).count()["Reason"].unstack()
plt.figure(figsize=(12, 6))
sea.heatmap(dh, cmap="viridis")
plt.show()

sea.clustermap(dh, cmap="viridis")
plt.show()

dm = df.groupby(by=["Day of Week", "Month"]).count()["Reason"].unstack()
plt.figure(figsize=(12, 6))
sea.heatmap(dm, cmap="viridis")
plt.show()

sea.clustermap(dm, cmap="viridis")
plt.show()
