'''

learning_rate= Modelin ağırlıklarını güncellerken atılacak adımın büyüklüğünü belirleyen bir parametre
gradient= Kayıp fonksiyonunun ağırlıklara göre türevini temsil eder ve modelin ağırlıklarının nasıl güncelleneceğini gösterir.
weights = learning_rate * gradient : modelin tahminlerini iyileştirir
L1 Reg. = modelin basitleşmesini sağlar.Ağırlıkların toplam mutlak değerini Cezalandırma (Penalization) eden bir regularizasyon yöntemidir
L2 Reg. = modelin aşırı uyumu önleyerek daha stabil bir model sağlar.Ağırlıkların karelerinin toplamını Cezalandırma (Penalization) eden bir regularizasyon yöntemidir
MSE = kayıp fonksiyonudur
Loss = tahminlerin ne kadar hatalı olduğunu gösterir.Modelin başarısını ölçer
BATCH_SIZE = mini-batch boyutu

Create Mini-Batch with iterations:
start_index ve end_index hesaplanarak, eğitim verisinden (X ve y) belirli bir aralıkta veriler alınır.
Bu, modelin her iterasyonda sadece veri kümesinin küçük bir alt kümesiyle çalışmasını sağlar, bu da genellikle hesaplama verimliliğini artırır ve daha hızlı bir öğrenme sağlar.

İterasyon Sayısı:
iterations_per_epoch, her epoch içinde kaç mini-batch olduğunu belirtir. Bu, eğitim verisinin toplam sayısına (n_samples) bağlıdır.
n_samples // batch_size ile hesaplanan bu değer, modelin toplam örnek sayısını mini-batch boyutuna böler ve her epoch için kaç kez güncelleme yapılacağını belirler.

mini_batch :
Modelin daha hızlı öğrenmesini sağlar.
Eğitim sırasında tüm veri kümesini kullanmak yerine, sadece bir kısmını kullanarak daha hızlı güncellemeler yapabilme imkanı sunar.

Mini-batch, eğitim veri setinden alınan bir alt küme,
Iterasyon ise bu mini-batch üzerinde yapılan bir güncelleme adımıdır.
Her iterasyonda bir mini-batch kullanılarak modelin ağırlıkları güncellenir.
Bu yüzden, bir epoch içindeki iterasyon sayısı, kullanılan mini-batch sayısına eşittir.

'''

import numpy as np


# Kayıp fonksiyonu (Mean Squared Error) tanımı
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def reg_l1(lambda_reg, weights):
    return lambda_reg * np.sum(np.abs(weights))


def reg_l2(lambda_reg, weights):
    return lambda_reg * np.sum(weights ** 2)


# L1 ve L2 regularizasyonu eklenmiş kayıp fonksiyonu
def loss_with_regularization(X, y, weights, lambda_reg, regularization_type='L1'):
    # Tahminleri hesaplama
    y_pred = X.dot(weights)

    # Kayıp hesaplama
    mse = mean_squared_error(y, y_pred)

    # Regularizasyon terimini ekleme
    if regularization_type == 'L1':
        regularization_term = reg_l1(lambda_reg, weights)
    elif regularization_type == 'L2':
        regularization_term = reg_l2(lambda_reg, weights)
    else:
        raise ValueError("Invalid regularization type. Use 'l1' or 'l2'.")

    return mse + regularization_term, mse


def model(X, weights):
    prediction = X.dot(weights)
    return prediction


# Gradient Descent ile model eğitimi
def train(X, y, lambda_reg, learning_rate=0.01, epochs=1000, regularization_type='L1'):
    weights = np.random.randn(X.shape[1])
    loss_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # Tahminleri hesapla
        y_pred = model(X, weights)

        # Hata hesapla
        error = y_pred - y

        # Gradient hesapla
        gradient = (2 / len(y)) * X.T.dot(error)

        # Regularizasyon gradienti ekle
        if regularization_type == 'L1':
            gradient += lambda_reg * np.sign(weights)  # L1 için mutlak değer türevi
        elif regularization_type == 'L2':
            gradient += 2 * lambda_reg * weights  # L2 için kare türevi

        # Ağırlıkları güncelle
        weights -= learning_rate * gradient

        # Kayıp fonksiyonunu kaydet
        loss, mse = loss_with_regularization(X, y, weights, lambda_reg, regularization_type)
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return weights, loss_history


def train_with_mini_batch(X, y, lambda_reg, learning_rate=0.01, epochs=1000, batch_size=10, regularization_type='L1'):
    weights = np.random.randn(X.shape[1])  # Başlangıç ağırlıkları
    loss_history = []

    n_samples = X.shape[0]  # Toplam örnek sayısı
    iterations_per_epoch = n_samples // batch_size  # Her epoch'daki iterasyon sayısı

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for iteration in range(iterations_per_epoch):
            # Mini-batch oluştur
            start_index = iteration * batch_size
            end_index = start_index + batch_size
            X_batch = X[start_index:end_index]
            y_batch = y[start_index:end_index]

            # Tahminleri hesapla
            y_pred = model(X_batch, weights)

            # Hata hesapla
            error = y_pred - y_batch

            # Gradient hesapla
            gradient = (2 / batch_size) * X_batch.T.dot(error)  # Gradient hesaplama

            # Regularizasyon gradienti ekle
            if regularization_type == 'L1':
                gradient += lambda_reg * np.sign(weights)  # L1 için mutlak değer türevi
            elif regularization_type == 'L2':
                gradient += 2 * lambda_reg * weights  # L2 için kare türevi

            # Ağırlıkları güncelle
            weights -= learning_rate * gradient  # Ağırlık güncelleme

            # Kayıp fonksiyonunu kaydet
            loss, mse = loss_with_regularization(X_batch, y_batch, weights, lambda_reg, regularization_type)
            loss_history.append(loss)

            # Her iterasyonda kaybı yazdır
            print(f"  Iteration {iteration + 1}/{iterations_per_epoch}, Loss: {loss:.4f}, MSE: {mse:.4f}")

    return weights, loss_history


# aynı başlangıç tohumu
np.random.seed(0)

n_samples = 100  # 1000 örnek
# n_features = 28 * 28  # 28x28 boyutunda resim
# X = np.random.rand(n_samples, n_features)  # Rastgele görsel veriler
# TRUE_WEIGHT = np.random.rand(n_features)
n_features = 10

X = np.random.rand(n_samples, n_features)  # 100 örnek, 10 özellik
TRUE_WEIGHT = np.array([3, 5, 0, 0, 0, 2, 0, 1, 0, 0])  # Gerçek ağırlıklar
NOISE = np.random.normal(0, 0.5, 100)  # ort. - standart sapma - 100 adet
y = X.dot(TRUE_WEIGHT) + NOISE  # Hedef değişken

EPOCHS = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 10

# L1 regularizasyonu ile model eğitimi
lambda_reg = 0.1  # Regularizasyon katsayısı
weights_l1, loss_history_l1 = train(X, y, lambda_reg, learning_rate=LEARNING_RATE, epochs=EPOCHS,
                                    regularization_type='L1')
# L2 regularizasyonu ile model eğitimi
weights_l2, loss_history_l2 = train(X, y, lambda_reg, learning_rate=LEARNING_RATE, epochs=EPOCHS,
                                    regularization_type='L2')

print("L1 Regularizasyonu ile Ağırlıklar:", weights_l1)
print("L2 Regularizasyonu ile Ağırlıklar:", weights_l2)

weights_l1_mini, loss_history_l1_mini = train_with_mini_batch(X, y, lambda_reg, learning_rate=LEARNING_RATE,
                                                              epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                              regularization_type='L1')

# L2 regularizasyonu ile model eğitimi
weights_l2_mini, loss_history_l2_mini = train_with_mini_batch(X, y, lambda_reg, learning_rate=LEARNING_RATE,
                                                              epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                              regularization_type='L2')

print("L1 Regularizasyonu ile Ağırlıklar + train_with_mini_batch:", weights_l1_mini)
print("L2 Regularizasyonu ile Ağırlıklar + train_with_mini_batch:", weights_l2_mini)
