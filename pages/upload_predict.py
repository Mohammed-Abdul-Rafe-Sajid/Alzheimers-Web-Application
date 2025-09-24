import streamlit as st
from utils.model_utils import load_model, preprocess_image, predict

st.header("Alzheimer's Detection from MRI Scan")

# Load model once
model = load_model("model/final_alz_model.pth", num_classes=4)
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# File uploader
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        image_tensor = preprocess_image(uploaded_file)
        prediction = predict(model, image_tensor, class_names)
        st.success(f"Prediction: **{prediction}**")
