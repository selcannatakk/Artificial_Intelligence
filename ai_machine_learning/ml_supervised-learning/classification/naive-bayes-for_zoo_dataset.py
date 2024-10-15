from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

data = pd.read_csv("hayvanatbahcesi.csv", encoding='unicode_escape')

input = np.array(data.drop(["sinifi"], axis=1))
output = np.array(data["sinifi"])

x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.35, random_state=109)

gnb = CategoricalNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
index = ['1', '2', '3', '4', '5', '6', '7']
columns = ['1', '2', '3', '4', '5', '6', '7']
cm_df = pd.DataFrame(cm, columns, index)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_df, annot=True, fmt="d")

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))