import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from visualization import cm_matrix


# Model mimarisini yükleme
with open("argo_model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Ağırlıkları yükleme
loaded_model.load_weights("argo_lstm_model.h5")

# Modeli kullanarak tahmin yapma
# Tahmin yapmak için giriş verisi (X) kullanmanız gerekecek
# Tokenize işlemi

texts = ["Bu selam jfhbvgf normal.",
         "Ne kadar argo bir cümle!",
         "Argo olmayan bir salak cümle.",
        ]
labels = np.array([0, 1, 0])
# print(len(texts[1]))  # geen girdi boyutunu alacak
vocab_size = 10000  # Varsayımsal kelime dağarcığı boyutu
max_len = 20
threshold = 0.5


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
text = pad_sequences(sequences, maxlen=max_len)

predictions = loaded_model.predict(text)
print(predictions)


# Tahminleri tam sayıya dönüştürme
rounded_predictions = [1 if pred >= threshold else 0 for pred in predictions]

cm_matrix(labels,rounded_predictions)
