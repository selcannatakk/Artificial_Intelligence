import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2

from custom_tf_cnn_model import SimpleNN, PretrainedResNetClassifier


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)

    return model


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_dir, class_to_index=None):
    images = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    class_to_index = {name: index for index, name in enumerate(classes)}

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = cv2.imread(file_path)  # image format = BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # #image format = RGB
            image = cv2.resize(image, (128, 128))

            images.append(np.array(image))  # Numpy dizisine ekle
            labels.append(class_to_index[class_name])

    return np.array(images), np.array(labels), class_to_index


def evaluate_model(model, test_images, test_labels, class_names, save_dir):
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    a = 0
    for i in range(len(test_images)):
        img = test_images[i] * 255.0  # normalize image for display
        img = img.astype(np.uint8)

        # Prediction ve gerçek etiketleri ekrana yazdır
        pred_class = class_names[predicted_classes[i]]
        true_class = class_names[true_classes[i]]
        label = f"Pred: {pred_class}, True: {true_class}"

        # img = cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        if pred_class == true_class:
            # save_path = os.path.join(save_dir+"/true", f"test_image_{i}_pred_{pred_class}_true_{true_class}.png")
            save_path = os.path.join(save_dir + "/true", f"{true_class}{a}.png")
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Save in BGR format for OpenCV
            cv2.imwrite(save_path, bgr_img)
        else:
            # save_path = os.path.join(save_dir + "/false", f"test_image_{i}_pred_{pred_class}_true_{true_class}.png")
            save_path = os.path.join(save_dir + "/false", f"{true_class}{a}.png")
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Save in BGR format for OpenCV
            cv2.imwrite(save_path, bgr_img)

        a += 1
        # cv2.imshow("Test Image", img)
        # cv2.waitKey(500)  # her görüntüyü 500 ms gösterir

    cv2.destroyAllWindows()


def main():
    config = load_config('config/config.yaml')

    train_images, train_labels, class_to_index = load_data(config['data']['train_data_dir'])
    val_images, val_labels, _ = load_data(config['data']['val_data_dir'])
    test_images, test_labels, _ = load_data(config['data']['test_data_dir'])

    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # one-hot encode
    num_classes = len(np.unique(train_labels))

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_labels = tf.keras.utils.to_categorical(val_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    model = SimpleNN(num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels,
                        epochs=config['training']['epochs'],
                        batch_size=config['training']['batch_size'],
                        validation_data=(val_images, val_labels))

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("./results/oc_flowers_accuracy_plot.png")

    model.save("./models/oc_custom_flowers_model", save_format="tf")

    model_path = "./models/oc_custom_flowers_model"

    model = load_model(model_path)

    class_names = {v: k for k, v in class_to_index.items()}

    save_dir = config['data']['save_dir']
    os.makedirs(save_dir + "/true", exist_ok=True)
    os.makedirs(save_dir + "/false", exist_ok=True)

    evaluate_model(model, test_images, test_labels, class_names, save_dir)


if __name__ == "__main__":
    main()
