import streamlit as st

st.title("About the Project")

st.write(
    """
    ###  Project Overview
    This project demonstrates how deep learning can assist in medical image analysis.  
    It uses a **Convolutional Neural Network (CNN)** built with **PyTorch** to classify MRI brain scans into 4 categories:  
    - Mild Demented  
    - Moderate Demented  
    - Non Demented  
    - Very Mild Demented  

    ### Dataset
    The model was trained on the publicly available [Alzheimer’s MRI dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset).  

    ### Technology Stack
    - **Python** (PyTorch, TorchVision, PIL)  
    - **Streamlit** for the website interface  
    - **GitHub** for version control  
    - **Streamlit Cloud** / **Hugging Face Spaces** for deployment  

    ### Future Improvements
    - Adding more advanced CNN architectures (ResNet, EfficientNet).  
    - Improving dataset diversity for higher accuracy.  
    - Detecting out-of-distribution (non-MRI) images.  
    """
)
