import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

# Görsel veriler için veri oluşturma (örneğin, 28x28 boyutunda resimler)
np.random.seed(0)
n_samples = 1000  # 1000 örnek
img_height, img_width, n_channels = 28, 28, 3  # Resim boyutu ve kanalları
X = np.random.rand(n_samples, img_height, img_width, n_channels)  # Rastgele görsel veriler
TRUE_WEIGHT = np.random.rand(img_height * img_width * n_channels)  # Gerçek ağırlıklar
NOISE = np.random.normal(0, 0.1, n_samples)  # Gürültü
y = np.dot(X.reshape(n_samples, -1), TRUE_WEIGHT) + NOISE  # Hedef değişken

# CNN Modelini tanımlama
def create_cnn_model(lambda_reg=0.01):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, n_channels),
                     kernel_regularizer=l2(lambda_reg)))  # İlk konvolüsyon katmanı
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Maksimum havuzlama
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(lambda_reg)))  # İkinci konvolüsyon katmanı
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Maksimum havuzlama
    model.add(Flatten())  # Düzleştirme
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(lambda_reg)))  # Tam bağlantılı katman
    model.add(Dense(1))  # Çıktı katmanı (regresyon)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Modeli derleme
    return model

# Modeli oluşturma
model = create_cnn_model(lambda_reg=0.01)

# Modeli eğitme
epochs = 100
batch_size = 32

# Modeli eğitme
history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Kayıp fonksiyonunun grafikle gösterimi
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.show()

# Ağırlıkları yazdırma
print("Model Ağırlıkları:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"{layer.name} Ağırlıkları: {weights}")
