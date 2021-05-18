import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

yelp = pd.read_csv("yelp.csv")

yelp["text length"] = yelp["text"].apply(len)

sns.set_style("white")
g = sns.FacetGrid(yelp, col="stars")
g.map(plt.hist, "text length", bins=50)
plt.show()

sns.boxplot(data=yelp, x="stars", y="text length", palette="rainbow")
plt.show()

sns.countplot(data=yelp, x="stars", palette="rainbow")
plt.show()

stars = yelp.groupby("stars").mean()
print(stars.corr())

sns.heatmap(stars.corr(), cmap="coolwarm", annot=True)
plt.show()

yelp_class = yelp[(yelp["stars"] == 1) | (yelp["stars"] == 5)]

X = yelp_class["text"]
y = yelp_class["stars"]

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

pipes = Pipeline([
    ("bow", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("classifier", MultinomialNB())
])

X = yelp_class["text"]
y = yelp_class["stars"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)
pipes.fit(X_train, y_train)

predictions = pipes.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
