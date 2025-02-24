import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

# Load Haar cascade and model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = joblib.load('face_recognition_model.pkl')

# Streamlit page config
st.set_page_config(page_title="Face Recognition App", page_icon="🤖", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
        body { font-family: 'Georgia', serif; }
        .stButton>button { background-color: #ff4b4b; color: white; font-size: 18px; border-radius: 8px; }
        .stButton>button:hover { background-color: #ff3333; }
        .stMarkdown { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Feature extraction function
def extract_features(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# App Title
st.title("🤖 Face Recognition System")
st.write("Upload an image or use your webcam to identify a person.")

# Image upload or capture
option = st.radio("Choose Input Method:", ["📸 Upload Image", "📷 Capture from Webcam"])

if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "📷 Capture from Webcam":
    captured_image = st.camera_input("Take a photo")
    if captured_image:
        image = Image.open(captured_image)

# Process image if available
if "image" in locals():
    col1, col2 = st.columns(2)

    # Display the selected image
    col1.image(image, caption="Uploaded/Captured Image", use_container_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detect face
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = img_array[y:y+h, x:x+w]

        # Draw bounding box
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
        col2.image(img_array, caption="Detected Face", use_container_width=True)

        # Extract features and predict
        features = extract_features(face_roi)
        if features is not None:
            with st.spinner("🔍 Identifying..."):
                prediction = model.predict([features])[0]
                confidence = model.predict_proba([features])[0].max() * 100

            # Display result
            st.success(f"✅ **Prediction:** {prediction} ({confidence:.2f}% confidence)")
        else:
            st.error("⚠️ Error processing face image.")
    else:
        st.warning("⚠️ No face detected!")
