import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def data_split(dataset):
  x_train, x_val, y_train, y_val = train_test_split(dataset.Message, dataset.Encoded_Label, test_size=0.2, random_state=0)
  return x_train, x_val, y_train, y_val


# dataset = pd.read_csv("../../../data/labeling_data.csv")
vocab_size = 10000  # Varsayımsal kelime dağarcığı boyutu
max_len = 20  # Maksimum cümle uzunluğu
embedding_dim = 50  # Gömme boyutu
epochs = 10
batch_size = 2


# x_train, x_val, y_train, y_val = data_split(dataset)

# vectorizer = CountVectorizer()
# x_train_count = vectorizer.fit_transform(x_train.values)
# x_train_array = x_train_count.toarray()


# texts = x_train[1]
texts = ["Bu cümle normal.",
         "Ne kadar argo bir cümle!",
         "Argo olmayan bir başka cümle.",
         "Burası çok pis bir yer.",
         "Dikkat, argo içerir!",
         "Sıradan bir ifade."
        ]
# label = x_train[2]
labels = np.array([0, 1, 0, 1, 1, 0])


# Metinleri sayısal diziye dönüştürme
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)

# -----------------------------------------------------------------

'''

LSTM MODEL 

'''
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=64))  # LSTM katmanı
model.add(Dense(units=1, activation='sigmoid'))  # Çıkış katmanı: 3 sınıf için -> softmax , 1 sınıf için -> sigmoid
model.summary()


# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# metrics = ["accuracy"]

'''

MODEL COMPPILER (derleme)

'''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
'''

MODEL TRAINING 

'''
history = model.fit(X, labels, epochs=epochs, batch_size=batch_size, verbose=1)
# model.fit(sequences, y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))

'''

MODEL EVALUATE

'''
# loss, accuracy = model.evaluate(X, labels, verbose=1)
history_metrics = model.evaluate(X, labels, verbose=1)
metrics_names = model.metrics_names
# print(f"Test verisi üzerinde doğruluk: {accuracy:.4f}")
print("Eğitim sürecinde elde edilen metrik değerler:")
for name, value in zip(metrics_names, history_metrics):
    print(f"{name}: {value}")
'''

MODEL SAVING 

'''
# Modelin ağırlıklarını ve mimarisini kaydetmek
model.save_weights("argo_lstm_model.h5")  # Ağırlıkları kaydetmek
with open("argo_model_architecture.json", "w") as json_file:
    json_file.write(model.to_json())  # Model mimarisini JSON formatında kaydetmek
