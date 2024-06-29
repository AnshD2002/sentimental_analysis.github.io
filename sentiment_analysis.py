import nltk
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import streamlit as st
import pickle
import subprocess
import sys
from nltk.corpus import stopwords


subprocess.call([sys.executable, "-m", "pip", "install", "nltk"])



# Function to remove stopwords from text
def remove_stopwords(text):
    if not isinstance(text, str):
        return ""

    stop_words = set(stopwords.words('english'))
    new_words = word_tokenize(text)
    new_filtered_words = [word for word in new_words if word.lower() not in stop_words]
    # Join the filtered words to form a clean text
    new_clean_text = ' '.join(new_filtered_words)
    return new_clean_text

# Function to remove punctuation from text
def remove_punctuation(text):
    if not isinstance(text, str):
        return ""

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

# Function to clean text by removing punctuation and stopwords
def clean_text(text):
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text

# Function to predict sentiment
def predict_sentiment(comment, model, tfidf):
    clean_comment = clean_text(comment)
    comment_list = [clean_comment]
    comment_vector = tfidf.transform(comment_list)
    comment_prediction = model.predict(comment_vector)[0]

    if comment_prediction == 1:
        sentiment = "Positive comment"
    else:
        sentiment = "Negative comment"

    return sentiment

# Streamlit interface
def main():
    st.title('Sentiment Analysis App')

    # User input area
    comment = st.text_area('Enter a comment')

    # Button for sentiment analysis
    if st.button('Analyze Sentiment'):
        # Load model and tfidf vectorizer
        try:
            with open('models/sentiment_model.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)

            with open('models/tfidf_vectorizer.pkl', 'rb') as tfidf_file:
                loaded_tfidf = pickle.load(tfidf_file)
        except FileNotFoundError:
            st.error("Model files not found. Please make sure the model files are in the 'models' directory.")
            return
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Predict sentiment if comment is provided
        if comment:
            sentiment = predict_sentiment(comment, loaded_model, loaded_tfidf)
            st.write('Sentiment:', sentiment)

            # Display images based on sentiment
            if sentiment == "Positive comment":
                st.image('images/smile.png', caption='Smiley Image')
            elif sentiment == "Negative comment":
                st.image('images/sad.png', caption='Sad Image')
        else:
            st.info("Enter a comment to analyze sentiment.")

if __name__ == '__main__':
    main()
