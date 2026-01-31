# Spam Email Detection System 📧

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spam-email-detection-system.streamlit.app/)

A machine learning project to detect spam emails using Naive Bayes (Multinomial & Gaussian). This system allows users to classify emails as **Spam** or **Ham** (Not Spam) through a Python script or an interactive Streamlit Web UI.

![Confusion Matrix](confusion_matrix_multinomial.png)

## 📌 Features
- **Data Preprocessing**: Tokenization, Stop-word removal, and Currency handling for financial spam detection.
- **Model Training**: Uses **Multinomial Naive Bayes** (best for text) and Gaussian Naive Bayes.
- **Data Augmentation**: Includes synthetic business spam examples (fake invoices, CEO fraud) to better detect modern threats.
- **Evaluation**: Generates Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.
- **Interactive UI**: A Streamlit-based web interface for real-time prediction.

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mugunthan-127/Spam-Email-Detection.git
cd Spam-Email-Detection
```

### 2. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit joblib fpdf
```

### 3. Download Dataset
The project requires the **SMS Spam Collection** dataset. Run the script to download it automatically:
```bash
python download_data.py
```

## 🛠 Usage

### Option A: Run via Command Line (Train & Evaluate)
To train the model, evaluate it, and generate reports:
```bash
python spam_detection.py
```
*   This creates `model_results.txt` and saves the trained models (`spam_model.joblib`).
*   It also generates performance plots (`confusion_matrix_multinomial.png`, etc.).

### Option B: Run the Web UI (Streamlit)
To start the interactive interface:
```bash
python -m streamlit run app.py
```
*   Opens in your browser at `http://localhost:8501`.
*   Type any email content to check if it's spam.
*   View performance metrics and download the **Final Project Report**.

### Option C: Generate PDF Report
To generate the `Final_Project_Report.pdf`:
```bash
python generate_report.py
```

## 📊 Performance
The **Multinomial Naive Bayes** model achieved the best results:
*   **Accuracy**: ~96.5%
*   **Precision (Spam)**: 100%
*   **Recall (Spam)**: ~74%

## 📂 Project Structure
*   `spam_detection.py`: Main script for training and evaluation.
*   `app.py`: Streamlit web application.
*   `download_data.py`: Downloads the required dataset.
*   `generate_report.py`: Generates the PDF project report.
*   `data/`: Contains the dataset.

## 👤 Author
**Mugunthan.M**
Register No: 727723EUCS127
