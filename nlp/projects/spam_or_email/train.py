import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


spam_dataset = pd.read_csv("../../../data/spam.csv")

spam_dataset.groupby('Category').describe()

spam_dataset['spam'] = spam_dataset['Category'].apply(lambda x: 1 if x == 'spam' else 0)
# print(spam_dataset)


x_train, x_val, y_train, y_val = train_test_split(spam_dataset.Message, spam_dataset.spam, test_size=0.2, random_state=0)

# x_train
print(x_train.describe())

vectorizer = CountVectorizer()
x_train_count = vectorizer.fit_transform(x_train.values)

print(x_train_count.toarray())

'''

MODEL

'''
model = MultinomialNB()
model.fit(x_train_count, y_train)

'''

TEST

'''
# test ham(email)
email_ham = [" hey wanna meet u for the game?"]
email_ham_count = vectorizer.transform(email_ham)
email_ham_count.toarray()
# 0 spam olmadığı anlamına gelir
model.predict(email_ham_count)

# test spam
email_spam = [" reward mony click"]
email_spam_count = vectorizer.transform(email_spam)
email_spam_count.toarray()
# 0 spam olmadığı anlamına gelir
model.predict(email_spam_count)

'''

TEST MODEL

'''
x_val_count = vectorizer.transform(x_val)
# x_test_count.toarray()
model.score(x_val_count, y_val)
