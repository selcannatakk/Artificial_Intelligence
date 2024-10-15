import os
import numpy as np
import pandas as pd
import PIL.Image as img
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn import metrics

covid = "../../../data/covid_veri-seti/covid/"
non_covid = "../../../data/covid_veri-seti/non_covid/"


def file(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)]


def data_transform(folder_name, class_name):
    images = file(folder_name)

    images_class = []
    for image in images:
        image_read = img.open(image).convert('L')
        image_resize = image_read.resize((28, 28))
        image_transform = np.array(image_resize).flatten()
        if class_name == "covid":
            datas = np.append(image_transform, [0])

        elif class_name == "non_covid":
            datas = np.append(image_transform, [1])

        else:
            continue
        images_class.append(datas)

    return images_class


covid_data = data_transform(covid, "covid")
covid_df = pd.DataFrame(covid_data)

non_covid_data = data_transform(non_covid, "non_covid")
non_covid_df = pd.DataFrame(non_covid_data)

tum_veri = pd.concat([covid_df, non_covid_df])

input = np.array(tum_veri)[:, :784]
output = np.array(tum_veri)[:, 784]
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=1)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
# MaxPool2D = görüntü boyutunu azaltmak için kullanılır
model.add(MaxPool2D(pool_size=(2, 2)))
# Dropout = overfiting engellemek için kullanılır
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# Flatten = katmanları sıkıstırarak tek boyutlu tek katman yapılır
model.add(Flatten())
# Dence = ileri beslenme sinir ağı için
model.add(Dense(256, activation="relu"))
model.add(Dropout(rate=0.5))
# softmax = calsfication (birden fazla class varsa)
model.add(Dense(43, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

clf = model.fit(x_train, y_train, epochs=2, batch_size=64)
y_pred = clf.predict(x_test)

print("Doğruluk:", metrics.accuracy_score(x_test, y_pred))

ypo, dpo, threshold = metrics.roc_curve(x_test, y_pred)
roc_auc = metrics.auc(ypo, dpo)
plt.title('ROC eğrisi')
plt.plot(ypo, dpo, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Doğru Pozitif Oranı')
plt.xlabel('Yanlış Pozitif Oranı')
plt.show()