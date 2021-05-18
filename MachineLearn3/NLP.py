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

# nltk.download_shell()

messages = [line.rstrip() for line in open("smsspamcollection/SMSSpamCollection")]

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)

messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep="\t", names=['label', "message"])
messages["length"] = messages["message"].apply(len)

messages["length"].plot.hist(bins=50)
plt.show()
messages.hist(column="length", by="label", bins=60, figsize=(12, 4))
plt.show()


def text_process(mess):
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [w for w in nopunc.split() if w.lower() not in stopwords.words("english")]


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages["message"])
print(len(bow_transformer.vocabulary_))

messages_bow = bow_transformer.transform(messages["message"])

print("shape of the Sparse Matrix:", messages_bow.shape)

tfidf_trans = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_trans.transform(messages_bow)

spam_detect = MultinomialNB().fit(messages_tfidf, messages["label"])

print(spam_detect.predict(messages_tfidf[4])[0])

msg_train, msg_test, la_train, la_test = train_test_split(messages["message"], messages["label"], test_size=.3)

pipes = Pipeline([
    ("bow", CountVectorizer(analyzer=text_process)),
    ("tfidf", TfidfTransformer()),
    ("classifier", MultinomialNB())
])

pipes.fit(msg_train, la_train)

predictions = pipes.predict(msg_test)
print(classification_report(la_test, predictions))
