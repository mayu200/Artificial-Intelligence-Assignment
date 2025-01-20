# Artificial Intelligence – Assignment  Document Classifier using NLP and Deep Learning

# Overview
# This project aims to build an intelligent document classifier that categorizes a set of documents into predefined classes using Natural Language Processing (NLP) and deep learning techniques. The task involved implementing and evaluating two models: Logistic Regression (Traditional Machine Learning) and LSTM (Deep Learning).

# Objectives
The goal is to classify documents into categories such as Scientific, Legal, E-commerce, and others using NLP techniques and machine learning.

# Preprocessing Steps
Data: 3000 text documents with three columns: ID, Text, and Category.
Cleaning: Tokenization, stop-word removal, and lemmatization.
Vectorization: TF-IDF representation with a maximum of 5000 features.
# Models Implemented
A. Logistic Regression
Implementation: L2 regularization with TF-IDF features.
Optimization: Grid Search to tune hyperparameters (C = 100, solver = 'liblinear').
Best Performance: F1-Score of 1.00.
B. LSTM
Architecture: Embedding layer, LSTM with 128 units, and Dense softmax layer for multiclass classification.
Performance: Struggled with an F1-Score of 0.078, indicating issues with input representation.
# Evaluation Results
Metric	Logistic Regression	LSTM
Accuracy	0.995	0.218
Precision	1.00	0.047
Recall	0.995	0.218
F1-Score	1.00	0.078 

# Challenges & Solutions
LSTM Performance: Replaced TF-IDF with pre-trained embeddings for future iterations.
Dimensionality: Limited TF-IDF features to 5000 terms.
Hyperparameter Tuning: Used Grid Search with cross-validation for Logistic Regression.
Conclusion
Logistic Regression performed excellently, while LSTM struggled, suggesting that better input representations are needed for deep learning models.
Future improvements include exploring embeddings like Word2Vec, GloVe, or BERT for LSTM.
