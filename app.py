import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time

# Set page config HARUS MENJADI PERINTAH STREAMLIT PERTAMA
st.set_page_config(
    page_title="🦎 Reptile Recognition",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set path untuk model dan assets
MODEL_PATH = os.path.join('models', 'Reptile.h5')

# Class names sesuai dengan urutan training
class_names = [
    "Crocodile/Alligator",
    "Frog",
    "Iguana",
    "Lizard",
    "Snake",
    "Turtle/Tortoise",
]

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        background-color: #4CAF50;
        transition: width 0.5s ease-in-out;
    }
    .header-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .upload-container {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
    }
    .species-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f1f8e9;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .sidebar-header {
        margin-bottom: 2rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(172, 172))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        probabilities = tf.nn.softmax(predictions[0])
        return probabilities.numpy()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Sidebar styling dan navigasi
with st.sidebar:
    st.markdown("<div class='sidebar-header'><h1>🦎 Reptile</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Recognition", "ℹ️ About"],
        index=0
    )

# Home page
if page == "🏠 Home":
    st.markdown("""
        <div class='header-container'>
            <h1>Welcome to Reptile Recognition AI</h1>
            <p>Discover and identify reptiles using our advanced AI system</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h3>🚀 How it works</h3>
                <ol>
                    <li>Upload a clear image of any reptile</li>
                    <li>Our Model analyzes the image</li>
                    <li>Get detailed classification results</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h3>🦎 Supported Species</h3>
            </div>
        """, unsafe_allow_html=True)
        for species in class_names:
            st.markdown(f"<div class='species-item'>{species}</div>", unsafe_allow_html=True)

# Recognition page
elif page == "🔍 Recognition":
    st.markdown("<h1 style='text-align: center;'>Reptile Recognition</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class='upload-container'>
                <h3>📸 Upload Image</h3>
                <p>Select a clear image of a reptile</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Image"):
                with st.spinner("🔄 Analyzing your image..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    probabilities = model_prediction(uploaded_file)
                    
                    if probabilities is not None:
                        with col2:
                            st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
                            
                            main_prediction_idx = np.argmax(probabilities)
                            main_prediction_prob = probabilities[main_prediction_idx]
                            
                            st.markdown(f"""
                                <div class='prediction-box'>
                                    <h3>🎯 Main Prediction</h3>
                                    <h2>{class_names[main_prediction_idx]}</h2>
                                    <p>Confidence: {main_prediction_prob:.2%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<h3>📈 All Probabilities</h3>", unsafe_allow_html=True)
                            
                            sorted_idx = np.argsort(probabilities)[::-1]
                            
                            for idx in sorted_idx:
                                prob = probabilities[idx]
                                st.markdown(f"""
                                    <div>
                                        <p>{class_names[idx]}: {prob:.2%}</p>
                                        <div class='confidence-bar'>
                                            <div class='confidence-fill' style='width: {prob*100}%;'></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

# About page
elif page == "ℹ️ About":
    st.markdown("<h1 style='text-align: center;'>About Our System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <h2>🤖 Model Information</h2>
            <ul>
                <li><strong>Architecture:</strong> MobileNetV2 with transfer learning</li>
                <li><strong>Input Size:</strong> 172x172 pixels</li>
                <li><strong>Training Method:</strong> Transfer Learning</li>
                <li><strong>Data Augmentation:</strong> Rotation, shift, zoom, and flip</li>
            </ul>
        </div>
        
        <div class='info-box'>
            <h2>📸 How to Get Best Results</h2>
            <ul>
                <li>Use clear, well-lit images</li>
                <li>Ensure the reptile is the main subject</li>
                <li>Avoid blurry or dark images</li>
                <li>Try different angles if unsure</li>
            </ul>
        </div>
        
        <div class='info-box'>
            <h2>🎯 Accuracy Tips</h2>
            <ul>
                <li>The model works best with front-facing shots</li>
                <li>Good lighting improves recognition accuracy</li>
                <li>Minimize background clutter</li>
                <li>Try to capture distinctive features</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)