# Sentiment Analysis
This project focuses on performing sentiment analysis on a combined dataset of Amazon product reviews and Twitter tweets. A total of 205,000 reviews and 50,083 tweets were extracted from an SQLite database and preprocessed to remove noise such as URLs, hashtags, mentions, non-English words, stopwords, and unnecessary characters. Special care was taken to convert emoticons, emojis, and numbers into meaningful word representations.

After cleaning, exploratory data analysis was conducted, including unigram and bigram frequency plots and word-length distributions to compare linguistic characteristics across platforms. The dataset was split into training, validation, and test sets while preserving sentiment class distribution. For feature engineering, TF-IDF vectors were created with unigrams and bigrams, and Word2Vec embeddings were trained to capture semantic context. 

For sentiment classification, three machine learning models were implemented:
- **Multinomial Logistic Regression** with L2 regularization  
- **Convolutional Neural Network (CNN)** with 400 filters and kernel size 5  
- **Bidirectional LSTM (Bi-LSTM)** with 64 and 32 units  

Model performance was evaluated using accuracy and F1-scores. The Bi-LSTM model achieved the highest accuracy at 72.8%, followed closely by CNN and Logistic Regression.

The entire project was developed in Python, leveraging libraries such as pandas, NLTK, scikit-learn, TensorFlow, and Matplotlib, with results documented in LaTeX.
