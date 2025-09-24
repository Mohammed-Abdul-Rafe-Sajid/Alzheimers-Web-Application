# Alzheimer’s Detection Web App
This is a Streamlit-based web application that uses a deep learning model (ResNet18 trained on grayscale MRI images) to classify Alzheimer’s disease stages.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the app
streamlit run app.py

---

## 📌 Next Steps for You
1. Create this folder structure on your system.  
2. Put the placeholder code in each file.  
3. Run it:
   ```bash
   streamlit run app.py
→ You’ll see a home page, an About page, and an Upload page (without ML yet).

4. Once that’s running smoothly, share your model file (.pth) with me.

I’ll check whether it was saved with state_dict or full model.

Then we’ll update upload_predict.py to load the model and actually give predictions.