import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Set path untuk model dan assets
MODEL_PATH = os.path.join('models', 'Reptile.h5')

# Class names sesuai dengan urutan training
class_names = [
    "Crocodile_Alligator",
    "Frog",
    "Iguana",
    "Lizard",
    "Snake",
    "Turtle_Tortoise",
]

# Fungsi untuk prediksi model
def model_prediction(test_image):
    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Preprocess image
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(172, 172))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0  # Normalisasi
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(image_array)
        probabilities = tf.nn.softmax(predictions[0])
        
        # Mengembalikan semua probabilitas
        return probabilities.numpy()
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Set page config
st.set_page_config(
    page_title="Reptile Recognition",
    page_icon="🦎",
    layout="wide"
)

# Sidebar untuk navigasi
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Reptile Recognition"])

# Halaman Home
if app_mode == "Home":
    st.header("Reptile Recognition System")
    st.markdown("""
    ### Welcome to Reptile Recognition System
    Our system helps identify different types of reptiles. Upload an image of a reptile, 
    and our AI system will analyze and classify it.
    Let's explore the fascinating world of reptiles!
    """)

# Halaman About
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Model
    This model uses MobileNetV2 architecture with transfer learning to classify reptiles.
    
    #### Model Details:
    - Input Image Size: 172x172 pixels
    - Base Model: MobileNetV2
    - Training Method: Transfer Learning
    - Data Augmentation: Rotation, shift, zoom, and flip
    
    #### Classes:
    {}
    """.format("\n".join([f"- {name}" for name in class_names])))

# Halaman Reptile Recognition
elif app_mode == "Reptile Recognition":
    st.header("Reptile Recognition")

    # Buat dua kolom
    col1, col2 = st.columns([1, 1])

    with col1:
        # Upload file
        test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if test_image is not None:
            st.image(test_image, caption="Uploaded Image", use_container_width=True)

    # Tombol untuk prediksi
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Analyzing image..."):
                probabilities = model_prediction(test_image)
                
                if probabilities is not None:
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Mendapatkan index dengan probabilitas tertinggi
                        result_index = np.argmax(probabilities)
                        max_probability = probabilities[result_index]
                        
                        # Menampilkan prediksi utama
                        st.markdown(f"### Main Prediction:")
                        st.markdown(f"**{class_names[result_index]}**")
                        st.markdown(f"Confidence: **{max_probability:.2%}**")
                        
                        # Menampilkan bar chart untuk semua probabilitas
                        st.markdown("### All Probabilities:")
                        
                        # Membuat dictionary untuk probabilitas
                        results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                        
                        # Mengurutkan hasil dari yang tertinggi ke terendah
                        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
                        
                        # Menampilkan bar chart
                        for animal, prob in sorted_results.items():
                            st.markdown(f"{animal}: {prob:.2%}")
                            st.progress(prob)
                        
        else:
            st.error("Please upload an image first")
