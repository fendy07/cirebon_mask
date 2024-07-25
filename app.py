import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('model/model-mask.h5', include_optimizer=True)
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004), metrics = ['accuracy'])

# Class names sesuai dengan dataset Anda
class_names = ['Klana', 'Pamindo', 'Panji', 'Rumyang', 'Tumenggung']

# Setup Streamlit
st.title('Image Classification - Topeng Cirebon')
st.write("Upload gambar topeng untuk diklasifikasikan:")

uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to a PIL image
    image = Image.open(uploaded_file)

    # Resize the image to the model input size
    image = image.resize((100, 100))

    # Convert the image to a numpy array
    image_array = img_to_array(image)

    # Add a batch dimension to the image
    image_array = np.expand_dims(image_array, axis=0)

    # Make predictions using the model
    predictions = model.predict(image_array)

    # Get the predicted class indices
    predicted_class_indices = np.argmax(predictions, axis=1)

    # Get the predicted class names
    predicted_class_names = []
    for i in predicted_class_indices:
        if 0 <= i < len(class_names):
            predicted_class_names.append(class_names[i])
        else:
            predicted_class_names.append("Unknown")

    # Display the predicted class names
    st.write("Predicted class:", predicted_class_names[0])

    # Display the uploaded image
    st.image(image, caption="Uploaded image")