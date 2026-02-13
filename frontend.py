import streamlit as st
import requests
from PIL import Image
import io

# ===== Page Config =====
st.set_page_config(
    page_title="Potato Disease Detection",
    layout="centered"
)

st.title("🌱 Potato Disease Detection System")
st.write("Upload leaf image to detect disease")

# ===== Image Upload =====
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button
    if st.button("Predict Disease"):
        with st.spinner("Analyzing Image..."):
            try:
                # API URL
                url = "http://127.0.0.1:8000/predict"

                files = {
                    "file": uploaded_file.getvalue()
                }

                response = requests.post(url, files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction Done ✅")

                    st.subheader(f"🧪 Disease: {result['class']}")
                    st.subheader(f"🎯 Confidence: {round(result['confidence']*100, 2)}%")

                    # Show all probabilities if available
                    if "all_probabilities" in result:
                        st.write("📊 All Probabilities:")
                        st.json(result["all_probabilities"])

                else:
                    st.error("API Error")

            except Exception as e:
                st.error(f"Error: {e}")
