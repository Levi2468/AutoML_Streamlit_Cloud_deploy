import streamlit as st

st.set_page_config(
    page_title="AutoML + AI Data Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AutoML Application")
st.subheader("Train ML Models &  Make Predictions")

st.write("""
---

## 🚀 What This Application Does

This platform allows you to:

✔ Upload any structured dataset (CSV / Excel)  
✔ Automatically preprocess data  
✔ Train multiple machine learning models  
✔ Select the best performing model  
✔ Make real-time predictions  
✔ Interact with your dataset using an AI assistant  

---

## 📤 page 1: Upload Dataset

Go to the **Upload** page and upload your dataset.

The application will:
- Preview your dataset
- Allow selection of target column
- Allow removal of unnecessary columns

---
## 🧹 page 2: Automatic Preprocessing

The system automatically:

- Detects **classification or regression**
- Handles missing values
- Encodes categorical features
- Removes duplicates
- Clips extreme outliers
- Detects class imbalance
- Applies **SMOTE safely (training data only)** if needed

---

## 🏋️ page 3:Model Training

The system trains optimized ML models including:

- Logistic / Linear Regression
- Decision Tree
- Random Forest
- XGBoost (lightweight configuration)

Evaluation Metrics:
- **ROC-AUC** (Classification)
- **R² Score** (Regression)

🏆 The best performing model is automatically selected and stored.

---

## 🎯 page 4: Prediction

After training:

- Dynamic input fields are generated
- Supports numerical & categorical inputs
- One-click prediction
- Displays predicted class/value
- Shows confidence score (for classification)
- Save & download prediction history
---
""")

