from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Örnek veri seti
texts = ["Bu bir pozitif cümle",
         "Bu bir negatif cümle",
         "Bu bir nötr cümle",
         "Başka bir pozitif örnek"]

labels = [1, 0, 2, 1]  # Etiketler: 1 (pozitif), 0 (negatif), 2 (nötr)

# Metinleri sayısal diziye dönüştürme
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Metin dizilerinin aynı uzunlukta olması için dolgu eklenmesi
maxlen = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=maxlen)

# Model oluşturma
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(LSTM(units=64))  # LSTM katmanı
model.add(Dense(units=3, activation='softmax'))  # Çıkış katmanı: 3 sınıf için softmax aktivasyonu

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(sequences, labels, epochs=10, batch_size=1, verbose=1)

# Modeli değerlendirme (opsiyonel)
# loss, accuracy = model.evaluate(sequences, labels)
# print('Loss:', loss)
# print('Accuracy:', accuracy)
