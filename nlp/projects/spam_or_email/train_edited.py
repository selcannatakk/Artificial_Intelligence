import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = "../../../data/spam.csv"


def labeling(dataset):
    #spam adında kolon olusturuyoruz
    spam_dataset['spam'] = dataset['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    return spam_dataset['spam']


def data_load(data):
    spam_dataset = pd.read_csv(data)
    spam_dataset.groupby('Category').describe() #change
    return spam_dataset


def dataset_split(dataset):
    x_train, x_val, y_train, y_val = train_test_split(dataset.Message, dataset.spam)
    return x_train, x_val, y_train, y_val


spam_dataset = data_load(data)

spam_dataset['spam'] = labeling(spam_dataset)

x_train, x_val, y_train, y_val = dataset_split(spam_dataset)
print(x_train)
print(x_train.describe())
