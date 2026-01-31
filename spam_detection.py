
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os
import joblib

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_data(filepath):
    # SMSSpamCollection is tab-separated
    df = pd.read_csv(filepath, sep='\t', names=['label', 'message'])
    print(f"Data loaded. Shape: {df.shape}")
    return df

def get_augmented_data():
    """
    Returns a DataFrame of business/finance spam examples to augment the dataset.
    This helps the model generalize to 'invoice fraud' and 'BEC' style spam.
    """
    data = [
        ('spam', 'Subject: Invoice #78421 – Payment Pending. Hi Team, Please find the attached invoice for last month’s services. Kindly process the payment before Friday to avoid late charges.'),
        ('spam', 'URGENT: Overdue payment notification. Your account #123456 is past due. Please pay immediately to avoid service suspension.'),
        ('spam', 'Attached is the outstanding invoice INV-999. Please remit payment by wire transfer to the account details below.'),
        ('spam', 'Dear Customer, your subscription has expired. Renew now to keep your services active. Click here to pay $49.99.'),
        ('spam', 'Payroll Department: Please review the attached salary slip and confirm your bank details urgently.'),
        ('spam', 'CEO Request: I need you to process a wire transfer of $5000 to our vendor immediately. Use the attached details.'),
        ('spam', 'Your order #5555 is pending. Finishing your payment of $200 is required to ship the items.'),
        ('spam', 'Final Reminder: Unpaid Invoice from last month. Amount Due: $1500. Pay via this link.'),
        ('spam', 'Security Alert: Suspicious activity on your bank account. Log in to verify your identity and avoid blocking.'),
        ('spam', 'Refund Notification: We owe you a refund of $300. Please fill out the form to claim it.'),
    ]
    return pd.DataFrame(data, columns=['label', 'message'])

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Handle Currency Symbols (preserve meaningful spam indicators)
    # Replace $ with 'dollarword', etc. to keep it as a feature after punctuation removal
    text = text.replace('$', ' dollarword ')
    text = text.replace('₹', ' rupee ')
    text = text.replace('€', ' euro ')
    
    # 3. Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # 4. Tokenize and remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)

def extract_features(corpus, method='tfidf'):
    if method == 'tfidf':
        # Added min_df to filter very rare words if needed, but keeping default for now
        vectorizer = TfidfVectorizer(max_features=5000) 
    elif method == 'bow':
        vectorizer = CountVectorizer(max_features=5000)
    else:
        raise ValueError("Method must be 'tfidf' or 'bow'")
    
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='multinomial'):
    if model_type == 'gaussian':
        model = GaussianNB()
        # GaussianNB requires dense matrix
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    elif model_type == 'multinomial':
        model = MultinomialNB()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='spam'),
        'recall': recall_score(y_test, y_pred, pos_label='spam'),
        'f1': f1_score(y_test, y_pred, pos_label='spam')
    }
    
    return model, y_pred, y_prob, metrics

def plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(y_test, y_prob, title, filename):
    # Convert labels to binary for ROC
    y_test_bin = y_test.map({'ham': 0, 'spam': 1})
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    data_path = os.path.join("data", "SMSSpamCollection")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run download_data.py first.")
        return

    print("Loading and Preprocessing Data...")
    df = load_data(data_path)
    
    # --- Data Augmentation ---
    print("Augmenting data with business spam examples...")
    augmented_df = get_augmented_data()
    df = pd.concat([df, augmented_df], ignore_index=True)
    print(f"New dataset shape: {df.shape}")
    # -------------------------
    
    # Preprocessing
    df['clean_text'] = df['message'].apply(preprocess_text)
    
    # Feature Extraction (TF-IDF)
    print("Extracting features (TF-IDF)...")
    X, vectorizer = extract_features(df['clean_text'], method='tfidf')
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Gaussian NB
    print("\nTraining Gaussian Naive Bayes...")
    gnb, gnb_pred, gnb_prob, gnb_metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model_type='gaussian')
    print("Gaussian NB Metrics:", gnb_metrics)
    plot_confusion_matrix(y_test, gnb_pred, "Confusion Matrix - Gaussian NB", "confusion_matrix_gaussian.png")
    if gnb_prob is not None:
        plot_roc_curve(y_test, gnb_prob, "ROC Curve - Gaussian NB", "roc_curve_gaussian.png")

    # Train Multinomial NB
    print("\nTraining Multinomial Naive Bayes...")
    mnb, mnb_pred, mnb_prob, mnb_metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model_type='multinomial')
    print("Multinomial NB Metrics:", mnb_metrics)
    plot_confusion_matrix(y_test, mnb_pred, "Confusion Matrix - Multinomial NB", "confusion_matrix_multinomial.png")
    if mnb_prob is not None:
        plot_roc_curve(y_test, mnb_prob, "ROC Curve - Multinomial NB", "roc_curve_multinomial.png")
    
    # Save Report Data
    with open("model_results.txt", "w") as f:
        f.write("Gaussian NB Metrics:\n")
        f.write(str(gnb_metrics) + "\n\n")
        f.write("Multinomial NB Metrics:\n")
        f.write(str(mnb_metrics) + "\n")
        
    print("\nExecution complete. Results saved.")

    # Save the Multinomial NB model and Vectorizer for Streamlit App
    print("Saving model and vectorizer...")
    joblib.dump(mnb, 'spam_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model saved as spam_model.joblib")
    print("Vectorizer saved as tfidf_vectorizer.joblib")

    # Real-World Applications: Demonstration
    print("\n--- Real-World Application Demonstration ---")
    sample_emails = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now.",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Please update your password immediately.",
        "Attached is the report you requested. Let me know if you have questions.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"
    ]
    
    # Use the best model (usually Multinomial for text)
    print("Classifying new emails using Multinomial NB model:")
    # Preprocess samples
    processed_samples = [preprocess_text(email) for email in sample_emails]
    # Transform
    X_samples = vectorizer.transform(processed_samples)
    # Predict
    predictions = mnb.predict(X_samples)
    
    for email, pred in zip(sample_emails, predictions):
        print(f"Email: '{email}'\n -> Prediction: {pred.upper()}\n")

if __name__ == "__main__":
    main()
