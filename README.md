# Sentiment Analysis: Logistic Regression vs. Neural Networks

**Authors:** Pooya Mers, Duy Phan, Robbert Bernecker, Erin Breur.

**Date:** June 2025.

**Course:** Data Science Project.

## Project Summary 
This project evaluates how traditional machine‑learning models compare to deep neural networks in **ternary sentiment classification** (positive, neutral, negative). Using a combined dataset of **205k Amazon reviews** and **50k Twitter tweets**, we build a full NLP pipeline, from data extraction and preprocessing to feature engineering, model training, and performance evaluation.

The goal is to determine whether neural networks meaningfully outperform logistic regression when applied to large, noisy, real‑world text data.

## Problem Statement 
Sentiment classification is widely used in industry to monitor customer opinions.
However, it remains unclear whether deep learning models (CNNs, LSTMs) consistently outperform simpler, more interpretable ones such as logistic regression, especially when trained on corpora like Amazon reviews and tweets.

This project answers three key questions:

1.**How do neural networks compare to logistic regression in accuracy and F1‑score?**

2.**Do richer text representations (embeddings) outperform TF‑IDF features?**

3.**How well do models adapt to two different text domains?**

## Approach

1. ## Data Preprocessing
- Extracted and merged Amazon and Twitter datasets from an SQLite database.
- Built a robust preprocessing pipeline including:
  - lowercasing, URL/mention/hashtag removal
  - emoji and emoticon mapping
  - punctuation and whitespace cleanup
  - selective stopword removal with negation handling
  - number‑to‑word conversion
- Conducted corpus analysis (unigrams, bigrams, word‑length distributions) to understand domain differences.
- Performed **stratified sampling** to maintain class balance across train/validation/test splits.

2. ## Feature Engineering
- Engineered two feature types:
  - **TF‑IDF vectors** (unigrams + bigrams) for logistic regression.
  - **Word2Vec embeddings** for neural networks.

3. ## Modeling Approaches
- Implementing and Evaluating Supervised ML models
  - Multinomial Logistic Regression (L2 regularization)
  - Convolutional Neural Network (CNN)
  - Bidirectional LSTM (Bi‑LSTM)
- Evaluated models using accuracy, macro‑F1, and weighted‑F1.

## Results
- Bi‑LSTM achieved the highest overall accuracy: 72.8%.
- CNN followed closely at 72.2%.
- Logistic Regression reached 71.4%, performing best on the neutral class.
- Neural networks benefited from word embeddings, capturing context beyond TF‑IDF.
- Logistic regression remained competitive, efficient, and highly interpretable.

## Key Takeaways
- Combining Amazon reviews and tweets introduces domain imbalance; neural networks handle this heterogeneity more effectively.
- Deep learning models provide a measurable performance gain, but logistic regression remains a strong baseline, especially for large, sparse TF‑IDF features.

## Skills:
- Data cleaning, NLP preprocessing, feature design, model development, and evaluation.

