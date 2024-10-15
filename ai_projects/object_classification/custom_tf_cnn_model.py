import tensorflow as tf



# Custom model class (Simple Neural Network) tanımlaması
class SimpleNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(SimpleNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3))  # image_height, image_width ve num_channels'ı tanımlaman gerekecek
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)


        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)

        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()


        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.batch_norm3(x)

        return self.output_layer(x)



class PretrainedResNetClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(PretrainedResNetClassifier, self).__init__()


        self.base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                         input_shape=(128, 128, 3))
        self.base_model.trainable = False

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_classes,
                                                  activation='softmax')  # Çıkış katmanı (3 sınıf: kedi, köpek, panda)

    def call(self, inputs):
        x = self.base_model(inputs, training=False)

        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)