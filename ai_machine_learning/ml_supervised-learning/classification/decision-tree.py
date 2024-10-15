from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
# bağımsız değişken(x)
print(iris.feature_names)
# bagımlı değişken(y)
print(iris.target_names)
print(iris.target)  # y
print(iris.data)  # x
X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print("Eğitim veri seti boyutu=", len(x_train))
print("Test veri seti boyutu=", len(x_test))

# model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_tahmin = model.predict(x_test)

# hata oranı görme
hata_matrisi = confusion_matrix(y_test, y_tahmin)
print(hata_matrisi)

# görselleştirme
index = ['setosa', 'versicolor', 'virginica']
columns = ['setosa', 'versicolor', 'virginica']
hata_goster = pd.DataFrame(hata_matrisi, columns, index)
plt.figure(figsize=(10, 6))
sns.heatmap(hata_goster, annot=True)