# Data Preprocessing Report

## Overview
This document outlines the text preprocessing steps applied to the SMS Spam Collection dataset to prepare it for Naive Bayes classification. Preprocessing is crucial to convert raw text into a format suitable for machine learning algorithms.

## Steps Implemented

### 1. Data Cleaning
- **Lowercasing**: All text was converted to lowercase to ensure uniformity (e.g., "Free" and "free" are treated as the same word).
- **Punctuation Removal**: All punctuation marks were removed. Punctuation often does not carry semantic meaning for spam detection and can add noise to the model.
  - *Method*: Python's `string.punctuation` library was used to filter out characters.

### 2. Tokenization
- The raw text strings were split into individual words (tokens). This is the fundamental step for Bag of Words and TF-IDF models.

### 3. Stop Word Removal
- **Stop Words**: Common English words (e.g., "the", "is", "at", "which") were removed. These words appear frequently but rarely contribute to determining whether an email is spam or ham.
- **Library**: `nltk.corpus.stopwords` (English list) was used.

### 4. Feature Extraction
To convert the text tokens into numerical features, we utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Bag of Words**.

#### TF-IDF Vectorization
- **Term Frequency (TF)**: Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF)**: Measures how important a term is. While computing TF, all terms are considered equally important. However, certain terms, such as "is", "of", and "that", may appear a lot but have little importance.
- **Implementation**: `sklearn.feature_extraction.text.TfidfVectorizer` was used. This converts the collection of raw documents to a matrix of TF-IDF features.

#### Bag of Words (BoW)
- Although TF-IDF was the primary method, the system supports Bag of Words via `CountVectorizer` for comparison if needed. BoW simply counts the occurrence of words.

## Conclusion
These preprocessing steps reduce the dimensionality of the data and focus the model on the meaningful keywords that likely differentiate spam from legitimate emails (e.g., "winner", "cash", "prize").
