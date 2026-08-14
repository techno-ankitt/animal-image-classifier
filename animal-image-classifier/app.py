import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/content/drive/MyDrive/animal-datasets/animal_classifier.h5')

model = load_model()

class_labels = ['Dog', 'Cat', 'Elephant', 'Horse', 'Butterfly', 'Spider']

st.title("🐾 Animal Image Classifier")
st.write("Upload an animal image and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
