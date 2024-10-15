import tensorflow as tf


class TextClassificationModel(tf.keras.Model):
    def __init__(self, encoder, vocab_size, embedding_dim=64, lstm_units=64):
        super(TextClassificationModel, self).__init__()
        self.encoder = encoder
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.embedding(x)
        x = self.bi_lstm(x)
        x = self.dense1(x)
        return self.dense2(x)

# # Örnek kullanım
# encoder = tf.keras.layers.experimental.utils.TextVectorization(max_tokens=1000)
# # Burada vocab_size'ı kendi verinizin kelime dağarcığı boyutuna ayarlamalısınız
# model = TextClassificationModel(encoder, vocab_size=1000)



# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         # Use masking to handle the variable sequence lengths
#         mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])