import streamlit as st
import joblib
import os
import string
from nltk.corpus import stopwords
import nltk
from PIL import Image

# Helper function for preprocessing (Must match the training script)
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Load Model and Vectorizer
@st.cache_resource
def load_assets():
    if not os.path.exists('spam_model.joblib') or not os.path.exists('tfidf_vectorizer.joblib'):
        return None, None
    model = joblib.load('spam_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Spam Email Detection System")
st.markdown("""
This application uses a **Multinomial Naive Bayes** machine learning model to detect spam emails. 
Enter an email content below to check if it's **Spam** or **Not Spam**.
""")

# Load assets
model, vectorizer = load_assets()

if model is None or vectorizer is None:
    st.error("Model files not found! Please run `spam_detection.py` first to train and save the model.")
else:
    # Sidebar for Report Download
    with st.sidebar:
        st.header("📄 Project Report")
        try:
            with open("Final_Project_Report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="Download Final Report (PDF)",
                    data=pdf_file,
                    file_name="Final_Project_Report.pdf",
                    mime="application/pdf"
                )
        except FileNotFoundError:
            st.warning("Report file not found. Please run `generate_report.py`.")

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Classifier", "📊 Performance Metrics"])

    with tab1:
        st.subheader("Real-time Classification")
        email_input = st.text_area("Enter Email Content", height=150, placeholder="Type or paste email text here...")
        
        if st.button("Predict"):
            if email_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                # Preprocess
                cleaned_text = preprocess_text(email_input)
                # Vectorize
                features = vectorizer.transform([cleaned_text])
                # Predict
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                if prediction == 'spam':
                    st.error(f"🚨 **SPAM DETECTED** (Confidence: {probability[1]:.2%})")
                else:
                    st.success(f"✅ **NOT SPAM (HAM)** (Confidence: {probability[0]:.2%})")
                    
                st.info(f"**Processed Text:** {cleaned_text}")

    with tab2:
        st.subheader("Model Performance Evaluation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Accuracy", value="96.53%")
            st.metric(label="Precision", value="100.00%")
        with col2:
            st.metric(label="Recall", value="74.11%")
            st.metric(label="F1 Score", value="85.13%")
            
        st.markdown("---")
        st.subheader("Visualizations")
        
        if os.path.exists("confusion_matrix_multinomial.png"):
            st.image("confusion_matrix_multinomial.png", caption="Confusion Matrix", use_column_width=True)
        else:
            st.warning("Confusion Matrix image not found.")
            
        if os.path.exists("roc_curve_multinomial.png"):
            st.image("roc_curve_multinomial.png", caption="ROC Curve", use_column_width=True)
        else:
            st.warning("ROC Curve image not found.")
