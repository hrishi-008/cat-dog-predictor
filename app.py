import streamlit as st
import joblib
import numpy as np
from keras.preprocessing import image

# Load the model
cnn_model = joblib.load('cnn_cat_dog.pkl')


# Streamlit app
def main():
    st.title("Cat or Dog Classifier")

    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the uploaded image for prediction
        processed_image = preprocess_image(uploaded_file)

        # Make prediction using the loaded model
        result = cnn_model.predict(processed_image)

        # Display the prediction result
        if result[0][0] == 1:
            prediction = 'Dog'
        else:
            prediction = 'Cat'

        st.success(f"The uploaded image is predicted to be a {prediction}.")


# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    # Load the image
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


if __name__ == "__main__":
    main()
