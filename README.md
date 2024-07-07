# Sentiment Analysis App

A sentiment analysis web application built with Python using Streamlit, NLTK for text preprocessing, and sklearn for machine learning.

## Overview

This application analyzes the sentiment of user-provided comments, categorizing them as positive or negative. It uses a machine learning model trained on a dataset to predict sentiment and displays corresponding images based on the sentiment detected.

### Features

- **Text Preprocessing:** Removes stopwords and punctuation from user comments.
- **Machine Learning Model:** Predicts sentiment using a pre-trained model.
- **Interactive UI:** Provides a user-friendly interface using Streamlit for input and result display.

## Files

- **sentiment_analysis.py**: Python script containing the Streamlit app code.
- **models/sentiment_model.pkl**: Pickled file containing the trained sentiment analysis model.
- **models/tfidf_vectorizer.pkl**: Pickled file containing TF-IDF vectorizer used for text transformation.
- **requirements.txt**: File listing dependencies required to run the application.

## Technologies Used

- Python
- Streamlit
- NLTK (Natural Language Toolkit)
- sklearn (scikit-learn)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-app.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run sentiment_analysis.py
