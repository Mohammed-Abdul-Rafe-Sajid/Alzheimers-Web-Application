import streamlit as st

st.set_page_config(
    page_title="Alzheimer's MRI Detection",
    page_icon="",
    layout="wide"
)

# Main Title
st.title("Alzheimer's MRI Detection Website")
st.markdown("### Welcome!")

st.write(
    """
    This website allows you to upload a brain MRI scan and get a prediction of possible Alzheimer's disease stage
    (Mild Demented, Moderate Demented, Non Demented, or Very Mild Demented).  
    The model is powered by **PyTorch** and trained on the [Alzheimer’s MRI dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset).
    """
)

st.markdown("---")

# Instructions
st.subheader("How to use this tool: ")
st.write(
    """
    1. Navigate to **Upload & Predict** from the sidebar.  
    2. Upload an MRI scan (JPG/PNG).  
    3. Click **Predict** to see the result.  
    4. View information about the project under **About**.  
    """
)

st.markdown("---")

# Links section
st.subheader(" Useful Links")
st.markdown(
    """
    - [View Source Code on GitHub](https://github.com/Mohammed-Abdul-Rafe-Sajid/Alzheimer-MRI-Disease-Classification-Deep-Learning)  
    - [Connect with me on LinkedIn](https://www.linkedin.com/in/mohammed-abdul-rafe-sajid-49a716291/)  
    - [MY Portfolio](https://mohammedabdulrafesajid.netlify.app/)
    """
)

st.markdown("---")

# Disclaimer
st.warning(
    """
    **Disclaimer**: This tool is for educational and demonstration purposes only.  
    It is **not a substitute for professional medical diagnosis**.  
    Please consult a medical expert for actual healthcare advice.
    """
)
