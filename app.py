import streamlit as st
import requests
import io
from PIL import Image

# --- Azure Custom Vision config ---
PREDICTION_KEY = "3r9SHrVjdXE5YlJop3bliQQvO0DUuO3pt3l1yVLBz5cBWCpz6XAKJQQJ99BEACYeBjFXJ3w3AAAIACOGUw5x"
PREDICTION_URL = "https://djhansirani-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/c652e6a6-e589-45a3-a8b9-b68739d9fd72/classify/iterations/dance-classifier-v1/image"

# Define headers and endpoint
headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}
endpoint = PREDICTION_URL

# --- Streamlit UI ---
st.set_page_config(page_title="Dance Classifier", layout="centered")
st.title("💃 Indian Dance Classification")
st.markdown("Upload a dance image and let AI classify the dance style!")

# Image uploader
uploaded_file = st.file_uploader("Upload an Indian dance image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and resize image for display
    image = Image.open(uploaded_file)
    fixed_width = 300
    w_percent = fixed_width / float(image.size[0])
    h_size = int(float(image.size[1]) * w_percent)
    resized_image = image.resize((fixed_width, h_size))
    st.image(resized_image, caption="Uploaded Image")

    # Predict button
    if st.button("Predict Dance Form"):
        # Convert image to byte stream
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()

        # Call Azure API
        response = requests.post(endpoint, headers=headers, data=image_bytes)

        # Process response
        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            top_prediction = predictions[0]

            st.success(f"🎯 Prediction: **{top_prediction['tagName']}** ({top_prediction['probability']*100:.2f}%)")

            # Optional: Show all predictions
            with st.expander("See all predictions"):
                for pred in predictions:
                    st.write(f"🔹 {pred['tagName']}: {pred['probability']*100:.2f}%")
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
else:
    st.info("Please upload an image to get started.")
