# Performance Evaluation Report

## Overview
This report documents the performance of the Naive Bayes classifiers (Gaussian and Multinomial) implemented for Spam Email Detection. The models were evaluated on a held-out test set (30% of the data).

## Metrics Defined
- **Accuracy**: The ratio of correctly predicted observations to the total observations.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. (Accuracy of positive predictions).
- **Recall**: The ratio of correctly predicted positive observations to the all observations in actual class. (Sensitivity).
- **F1 Score**: The weighted average of Precision and Recall.

## Model Performance

### 1. Gaussian Naive Bayes
- **Accuracy**: 89.89%
- **Precision**: 57.83%
- **Recall**: 90.62%
- **F1 Score**: 70.61%

**Confusion Matrix**:
![Confusion Matrix - Gaussian](confusion_matrix_gaussian.png)

**ROC Curve**:
![ROC Curve - Gaussian](roc_curve_gaussian.png)

### 2. Multinomial Naive Bayes
- **Accuracy**: 96.53%
- **Precision**: 100.00%
- **Recall**: 74.11%
- **F1 Score**: 85.13%

**Confusion Matrix**:
![Confusion Matrix - Multinomial](confusion_matrix_multinomial.png)

**ROC Curve**:
![ROC Curve - Multinomial](roc_curve_multinomial.png)

## Comparison & Analysis
- **Multinomial Naive Bayes** significantly outperformed Gaussian NB in terms of Accuracy (96.53% vs 89.89%) and Precision (100% vs 57.83%).
- **Gaussian Naive Bayes** had higher Recall (90.62%), meaning it caught more spam, but at the cost of flagging many legitimate emails as spam (low precision).
- **Multinomial Naive Bayes** had perfect precision, meaning if it called something spam, it was definitely spam. This is often preferred in email filters to avoid moving important emails to the spam folder.

**Observation**:
The Multinomial model is the better choice for this application due to its high accuracy and zero false positive rate (Precision = 1.0).

## Conclusion
The developed system effectively detects spam emails. The **Multinomial Naive Bayes** model performed best with an accuracy of **96.53%**.
