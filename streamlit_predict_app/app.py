import streamlit as st
import cv2
import numpy as np

from utils import load_model


def predict_for_image(model, uploaded_image):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Görsel yüklenemedi. Lütfen geçerli bir resim dosyası yükleyin.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    image = cv2.resize(image, (128, 128)) # model input size
    img_array = image.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # st.write(img_array)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_index


def main():
    model_path = "../streamlit_predict_app/models/oc_custom_flowers_model"

    model = load_model(model_path)

    st.title("🌼 Çiçek Tanıma Asistanı 🌼")
    st.write("Lütfen sınıflandırmak istediğiniz çiçek görselini yükleyin. ")

    uploaded_file = st.file_uploader("Bir çiçek görseli yükleyin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        class_index = predict_for_image(model, uploaded_file)

        classes = ["DAISY", "ROSES", "SUNFLOWERS"]
        st.write(f"**  Tahmin Edilen Çiçek Türü **: {classes[class_index]} ")



if __name__ == '__main__':
    main()
