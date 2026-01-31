from fpdf import FPDF
import os

class ProjectReport(FPDF):
    def header(self):
        # Page Border
        self.set_draw_color(0, 0, 0) # Black
        self.set_line_width(0.5)
        self.rect(5.0, 5.0, 200.0, 287.0) # A4 size approx 210x297 minus margins
        
        # Header Text (only on non-title pages)
        if self.page_no() > 1:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(100, 100, 100) # Gray
            self.cell(0, 10, 'SPAM EMAIL DETECTION - FINAL REPORT', 0, 1, 'R')
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 51, 102) # Dark Blue
        self.cell(0, 10, f'{num}. {label}', 0, 1, 'L')
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 200, self.get_y()) # Underline
        self.ln(5)
        self.set_text_color(0, 0, 0) # Reset to black

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

    def add_image(self, image_path, title):
        if os.path.exists(image_path):
            self.ln(5)
            self.set_font('Arial', 'B', 10)
            self.set_text_color(0, 0, 0)
            self.cell(0, 10, title, 0, 1, 'C')
            # Adjust width to fit page
            self.image(image_path, x=20, w=170) 
            self.ln(5)
        else:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"[Image {title} not found]", 0, 1, 'C')

def create_report():
    pdf = ProjectReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- Title Page ---
    # Double Border for Title Page
    pdf.set_draw_color(0, 51, 102)
    pdf.set_line_width(1)
    pdf.rect(10, 10, 190, 277)
    
    pdf.ln(60)
    pdf.set_text_color(0, 51, 102) # Dark Blue
    pdf.set_font('Arial', 'B', 24)
    pdf.multi_cell(0, 10, 'SPAM EMAIL DETECTION\nUSING NAIVE BAYES', 0, 'C')
    pdf.ln(10)
    
    pdf.set_draw_color(0, 0, 0)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(20)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'PROJECT FINAL REPORT', 0, 1, 'C')
    pdf.ln(30)
    
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Submitted by:', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Mugunthan.M', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Register No: 727723EUCS127', 0, 1, 'C')
    
    # Add GitHub Link to Title Page
    pdf.ln(20)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Project Repo:', 0, 1, 'C')
    pdf.set_font('Arial', 'U', 12)
    pdf.set_text_color(0, 0, 255)
    pdf.cell(0, 10, 'https://github.com/Mugunthan-127/Spam-Email-Detection', 0, 1, 'C', link='https://github.com/Mugunthan-127/Spam-Email-Detection')
    
    pdf.add_page()
    
    # 1. Introduction
    pdf.chapter_title('1', 'INTRODUCTION')
    pdf.chapter_body("""Email communication has become an essential part of modern digital communication. However, the increasing number of spam emails has created serious issues such as information overload, phishing attacks, and data theft. Spam emails often contain malicious links, fake offers, or misleading information that can cause financial and security risks.

To overcome this problem, machine learning techniques can be used to automatically classify emails as spam or non-spam (ham). This project focuses on implementing a Spam Email Detection System using Naive Bayes Classification. The system analyzes the content of emails and predicts whether an email is spam or legitimate based on learned patterns.""")

    # 2. Objectives
    pdf.chapter_title('2', 'OBJECTIVES OF THE PROJECT')
    pdf.chapter_body("""The main objectives of this project are:
- To design a spam email detection system using machine learning.
- To apply text preprocessing techniques such as tokenization and stop-word removal.
- To extract features using Bag of Words (BoW) and TF-IDF.
- To implement Multinomial Naive Bayes and Gaussian Naive Bayes.
- To evaluate accuracy, precision, recall, and F1-score.
- To test the model on unseen real-world email samples.""")

    # 3. Dataset
    pdf.chapter_title('3', 'DATASET DESCRIPTION')
    pdf.chapter_body("""The dataset consists of email messages labeled as:
0 -> Not Spam (Ham)
1 -> Spam

It is balanced and includes both genuine and spam emails.
Example: "Win 10,000 now" (Spam), "Meeting at 10 AM" (Ham).""")

    # 4. Preprocessing
    pdf.chapter_title('4', 'DATA PREPROCESSING')
    pdf.chapter_body("""Preprocessing steps:
- Lowercasing: Converts text to lowercase.
- Currency Handling: Preserves "$" and other symbols as features.
- Tokenization: Splits sentences into words.
- Stop Word Removal: Removes common words like 'is', 'the'.
- Feature Extraction: Uses TF-IDF and Bag of Words.""")

    # 5. Methodology
    pdf.chapter_title('5', 'METHODOLOGY')
    pdf.chapter_body("""5.1 Bag of Words (BoW): Counts word occurrences. Used with Multinomial NB.
5.2 TF-IDF: Assigns importance to rare words. Used with Gaussian NB.""")

    # 6. Models
    pdf.chapter_title('6', 'MACHINE LEARNING MODELS USED')
    pdf.chapter_body("""- Multinomial Naive Bayes: Best for word counts/text data.
- Gaussian Naive Bayes: Assumes normal distribution, useful for continuous features.""")

    # 7. Implementation
    pdf.chapter_title('7', 'IMPLEMENTATION')
    pdf.chapter_body("Steps: Load -> Preprocess -> Split -> Extract Features -> Train -> Predict -> Evaluate.")

    # 8. Performance Evaluation (With PLACEHOLDERS for visuals)
    pdf.chapter_title('8', 'PERFORMANCE EVALUATION')
    pdf.chapter_body("""The models were evaluated using Accuracy, Precision, Recall, and F1-Score.

Sample Results (Approximate):
Multinomial NB: ~96% Accuracy, 100% Precision (Spam)
Gaussian NB: ~90% Accuracy, ~58% Precision (Spam)

Observation: Multinomial Naive Bayes performs significantly better for this text classification task.""")

    # 9. Confusion Matrix (Visuals First)
    pdf.add_page()
    pdf.chapter_title('9', 'CONFUSION MATRIX & ROC CURVES')
    
    # Add Images
    pdf.add_image('confusion_matrix_multinomial.png', 'Confusion Matrix - Multinomial NB')
    pdf.add_image('roc_curve_multinomial.png', 'ROC Curve - Multinomial NB')
    
    pdf.chapter_body("""The Confusion Matrix visualizes True Positives, True Negatives, False Positives (Ham classified as Spam), and False Negatives (Spam classified as Ham).
The ROC Curve illustrates the diagnostic ability of the classifier.""")

    # 10. Real World Apps
    pdf.chapter_title('10', 'REAL-WORLD APPLICATIONS')
    pdf.chapter_body("- Email spam filters\n- Phishing detection\n- Corporate security")

    # 11. Limitations
    pdf.chapter_title('11', 'LIMITATIONS')
    pdf.chapter_body("- May miss highly obfuscated spam.\n- Requires constant retraining.")

    # 12. Future
    pdf.chapter_title('12', 'FUTURE ENHANCEMENTS')
    pdf.chapter_body("- Deep Learning (BERT/LSTM)\n- URL/Sender Analysis")

    # 13. Conclusion
    pdf.chapter_title('13', 'CONCLUSION')
    pdf.chapter_body("""In this project, a Spam Email Detection System was developed. Multinomial Naive Bayes proved to be the most effective model, achieving high accuracy and precision. The inclusion of currency symbol tracking and business spam augmentation improved the detection of financial scams.""")

    # Appendix: Code
    pdf.add_page()
    pdf.chapter_title('APPENDIX', 'SOURCE CODE (spam_detection.py)')
    pdf.set_font('Courier', '', 8)
    try:
        with open('spam_detection.py', 'r', encoding='utf-8') as f:
            code_lines = f.readlines()
            for line in code_lines:
                # Replace unicode characters that FPDF's standard fonts can't handle (latin-1 issue)
                line = line.replace('₹', 'Rs.').replace('€', 'Euro').replace('’', "'").replace('–', '-')
                # Remove any other non-latin-1 chars
                line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 4, line.replace('\t', '    '))
    except FileNotFoundError:
         pdf.multi_cell(0, 5, "Source code file not found.")

    pdf.output('Final_Project_Report.pdf')
    print("PDF Report generated: Final_Project_Report.pdf")

if __name__ == "__main__":
    create_report()
