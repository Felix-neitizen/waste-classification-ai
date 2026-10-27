
from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="centered"
)

# Project paths
project = Path(__file__).parent
model_path = project / "waste_classifier.keras"

# Page title
st.title("♻️ Waste Classification AI")
st.write(
    "Upload an image to classify it as plastic, paper, or cardboard."
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# Upload an image
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded image", width=350)

        if st.button("Classify Image"):
            with st.spinner("Classifying image..."):
                image = image.resize((160, 160))
                image_array = np.array(image, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]

                class_names = ["cardboard", "paper", "plastic"]
                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index])

            st.success(f"Predicted category: {predicted_class.title()}")
            st.metric("Model confidence", f"{confidence:.2%}")

            st.subheader("Scores for each category")
            for index, name in enumerate(class_names):
                st.write(f"**{name.title()}**: {predictions[index]:.2%}")
                st.progress(float(predictions[index]))

    except Exception as error:
        st.error(f"Could not process this image: {error}")
