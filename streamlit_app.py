import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to extract data into a DataFrame
def extract_data(text):
    lines = text.splitlines()
    data = [{"Line": line.strip()} for line in lines if line.strip()]
    return pd.DataFrame(data)

# Function to search for keywords
def search_keywords(text, keywords):
    results = {keyword: len(re.findall(keyword, text, re.IGNORECASE)) for keyword in keywords}
    return results

# Function to summarize text
def summarize_text(text):
    return "\n".join(text.splitlines()[:5])

# Function to clean text
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to count words
def count_words(text):
    return len(text.split())

# Function to count sentences
def count_sentences(text):
    return len(re.split(r'[.!?]+', text)) - 1

# Function to count characters
def count_characters(text):
    return len(text)

# Function to calculate reading time
def calculate_reading_time(text):
    words = count_words(text)
    reading_speed = 200  # average reading speed (words per minute)
    return round(words / reading_speed, 2)

# Function to extract links
def extract_links(text):
    return re.findall(r'https?://\S+', text)

# Function to extract emails
def extract_emails(text):
    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

# Function to find phone numbers
def find_phone_numbers(text):
    return re.findall(r'\+?\d[\d -]{8,12}\d', text)

# Function to extract dates
def extract_dates(text):
    return re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)

# Function to extract unique words
def extract_unique_words(text):
    words = text.split()
    unique_words = set(words)
    return list(unique_words)

# Function to calculate sentiment of text
def calculate_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to extract top N keywords using TF-IDF
def extract_top_n_keywords(text, n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    top_n_indices = np.argsort(tfidf_scores)[-n:][::-1]
    return {feature_names[i]: tfidf_scores[i] for i in top_n_indices}

# Function to count paragraphs
def count_paragraphs(text):
    return len([p for p in text.split('\n\n') if p.strip()])

# Function to calculate frequency distribution
def frequency_distribution(text):
    words = text.split()
    return Counter(words).most_common(10)

# Function to highlight keywords
def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = re.sub(f'({keyword})', r'**\1**', text, flags=re.IGNORECASE)
    return text

# Function to calculate readability
def calculate_readability(text):
    words = count_words(text)
    sentences = count_sentences(text)
    paragraphs = count_paragraphs(text)
    
    if sentences > 0:
        readability_score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (paragraphs / sentences))
    else:
        readability_score = float('inf')
    
    return readability_score

# Streamlit UI
st.title("Advanced PDF Data Extraction and Analysis")
st.write("Upload a PDF file to extract and analyze data.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Text")
    st.write(text)

    data_df = extract_data(text)
    st.subheader("Extracted Data")
    st.dataframe(data_df)

    keywords_input = st.text_input("Enter keywords to search (comma-separated):")
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',')]
        keyword_results = search_keywords(text, keywords)
        st.subheader("Keyword Search Results")
        st.write(keyword_results)
        highlighted_text = highlight_keywords(text, keywords)
        st.subheader("Highlighted Text")
        st.write(highlighted_text)

    st.subheader("Text Summary")
    st.write(summarize_text(text))

    st.subheader("Cleaned Text")
    st.write(clean_text(text))

    st.subheader("Text Statistics")
    stats = {
        "Word Count": count_words(text),
        "Character Count": count_characters(text),
        "Sentence Count": count_sentences(text),
        "Paragraph Count": count_paragraphs(text),
        "Unique Words": len(extract_unique_words(text)),
        "Reading Time (minutes)": calculate_reading_time(text),
        "Readability Score": calculate_readability(text),
        "Sentiment Polarity": calculate_sentiment(text)
    }
    st.write(stats)

    st.subheader("Extracted Links")
    st.write(extract_links(text))

    st.subheader("Extracted Emails")
    st.write(extract_emails(text))

    st.subheader("Extracted Phone Numbers")
    st.write(find_phone_numbers(text))

    st.subheader("Extracted Dates")
    st.write(extract_dates(text))

    st.subheader("Top N Keywords")
    top_n_keywords = extract_top_n_keywords(text)
    st.write(top_n_keywords)

    st.subheader("Frequency Distribution of Words")
    freq_dist = frequency_distribution(text)
    st.write(freq_dist)

    st.subheader("Unique Words")
    st.write(extract_unique_words(text))

    st.subheader("Extracted Dates")
    st.write(extract_dates(text))

    st.subheader("Sentence Splitting")
    sentences = re.split(r'(?<=[.!?]) +', text)
    st.write(sentences)

    st.subheader("Extracted Tables")
    st.write("Table extraction not implemented.")

    st.subheader("Word Cloud")
    st.write("Word cloud generation not implemented.")

    csv = data_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download extracted data as CSV",
        data=csv,
        file_name='extracted_data.csv',
        mime='text/csv'
    )
