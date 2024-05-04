import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Adjust based on your model architecture

# Load the pre-trained model
model_path = 'keras_model.h5'  # Replace with the path to your .h5 file
model = load_model(model_path)

# Class labels (replace with your own class labels)
class_labels = ["Real", "Fake"]

# Streamlit app
def main():
    st.title("Detection of Counterfeit shoes")

    # User upload for shoe image
    uploaded_file = st.file_uploader("Choose a shoe image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the model
        img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjust based on your model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Adjust based on your model's preprocessing

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0, predicted_class]

        st.success(f"The shoe is detected as: {class_labels[predicted_class]}")

if __name__ == '__main__':
    main()
