import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("College_Data", index_col=0)

sns.lmplot(data=df, x="Room.Board", y="Grad.Rate", hue="Private", fit_reg=False, palette="coolwarm", height=6, aspect=1)
plt.show()

sns.lmplot(data=df, x="Outstate", y="F.Undergrad", hue="Private", fit_reg=False, height=6, aspect=1)
plt.show()

g = sns.FacetGrid(df, hue="Private", palette="coolwarm", height=6, aspect=2)
g = g.map(plt.hist, "Outstate", bins=20, alpha=.7)
plt.show()

g = sns.FacetGrid(df, hue="Private", palette="coolwarm", height=6, aspect=2)
g = g.map(plt.hist, "Grad.Rate", bins=20, alpha=.7)
plt.show()

print(df[df["Grad.Rate"] > 100])
df["Grad.Rate"]["Cazenovia College"] = 100

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop("Private", axis=1))


def converter(cluster):
    if cluster == "Yes":
        return 1
    else:
        return 0


df["Cluster"] = df["Private"].apply(converter)

print(confusion_matrix(df["Cluster"], kmeans.labels_))
print(classification_report(df["Cluster"], kmeans.labels_))
