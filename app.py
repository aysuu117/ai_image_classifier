import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Image Classification System")
st.write("Upload an image and let Artificial Intelligence recognize it.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("AI is analyzing the image..."):
            model = load_model()

            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image)

            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)

            predictions = model.predict(image_array, verbose=0)
            results = decode_predictions(predictions, top=5)[0]

            labels = [item[1].replace("_", " ").title() for item in results]
            scores = [float(item[2]) * 100 for item in results]

        st.success("Analysis completed!")

        st.subheader("Top Prediction")
        st.write(f"### {labels[0]}")
        st.write(f"Model score: {scores[0]:.2f}%")

        st.subheader("Top 5 Predictions")

        df = pd.DataFrame({
            "Object": labels,
            "Score (%)": [round(score, 2) for score in scores]
        })

        st.dataframe(df, use_container_width=True)

        st.bar_chart(
            df.set_index("Object")["Score (%)"]
        )

st.divider()
st.caption("Powered by MobileNetV2 and TensorFlow")