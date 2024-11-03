import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from bs4 import BeautifulSoup
import numpy as np
import re
from wordcloud import WordCloud
from collections import Counter
from textstat import flesch_reading_ease

# Title and file uploader on the main page
st.title("Document Analysis App")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

# Function to extract text from various formats
def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "text/html":
        return extract_text_from_html(file)

# PDF extraction
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# DOCX extraction
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# HTML extraction
def extract_text_from_html(file):
    soup = BeautifulSoup(file.read(), 'html.parser')
    return soup.get_text()

# Text Cleaning
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Function to calculate text statistics
def character_count(text):
    return len(text)

def word_count(text):
    return len(text.split())

def sentence_count(text):
    return text.count('.') + text.count('!') + text.count('?')

def longest_word(text):
    words = text.split()
    return max(words, key=len) if words else ''

def shortest_word(text):
    words = text.split()
    return min(words, key=len) if words else ''

def average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0

def text_statistics(text):
    return {
        "Character Count": character_count(text),
        "Word Count": word_count(text),
        "Sentence Count": sentence_count(text),
        "Longest Word": longest_word(text),
        "Shortest Word": shortest_word(text),
        "Average Word Length": average_word_length(text)
    }

# Simple sentiment analysis
def simple_sentiment_analysis(text):
    words = text.split()
    positive_words = set(['good', 'great', 'excellent', 'positive', 'fortunate', 'correct', 'superior'])
    negative_words = set(['bad', 'poor', 'terrible', 'negative', 'unfortunate', 'wrong', 'inferior'])
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    return positive_count - negative_count

# Summarization
def summarize_text(text, num_sentences=3):
    sentences = text.split('. ')
    return '. '.join(sentences[:num_sentences])

# Keyword Extraction using TF-IDF
def keyword_extraction_tfidf(text):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=feature_names).T.sort_values(0, ascending=False).head(10)

# Generate Word Cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Email Extraction
def extract_emails(text):
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    return re.findall(email_pattern, text)

# Link Extraction
def extract_links(text):
    return re.findall(r'https?://\S+', text)

# Frequency Distribution of Words
def word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    return word_counts.most_common(10)

# Visualizing Word Frequency
def plot_word_frequency(word_counts):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.xticks(rotation=45)
    plt.title('Top 10 Most Frequent Words')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Main application logic
if uploaded_files:
    all_texts = []
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        all_texts.append(text)

    full_text = "\n\n".join(all_texts)
    
    # Text Analysis Section
    st.subheader("Extracted Text")
    st.write(full_text)

    # Cleaned Text
    cleaned_text = clean_text(full_text)
    st.subheader("Cleaned Text")
    st.write(cleaned_text)

    # Text Statistics
    stats = text_statistics(cleaned_text)
    st.subheader("Text Statistics")
    st.json(stats)

    # Simple Sentiment Analysis
    sentiment_score = simple_sentiment_analysis(cleaned_text)
    st.subheader("Sentiment Analysis")
    st.write(f"Sentiment Score: {sentiment_score}")

    # Summary of Text
    st.subheader("Text Summary")
    summary = summarize_text(cleaned_text)
    st.write(summary)

    # Keyword Extraction using TF-IDF
    st.subheader("Keyword Extraction")
    df_tfidf = keyword_extraction_tfidf(cleaned_text)
    st.write(df_tfidf)

    # Generate Word Cloud
    st.subheader("Keyword Cloud")
    generate_word_cloud(cleaned_text)

    # Email and Link Extraction
    st.subheader("Extracted Emails")
    emails = extract_emails(cleaned_text)
    st.write(emails)
    st.write(f"Total Emails: {len(emails)}")

    st.subheader("Extracted Links")
    links = extract_links(cleaned_text)
    st.write(links)
    st.write(f"Total Links: {len(links)}")

    # Frequency Distribution
    st.subheader("Top 10 Most Frequent Words")
    freq_words = word_frequency(cleaned_text)
    plot_word_frequency(freq_words)

    # Download processed text if needed
    st.sidebar.subheader("Download Processed Text")
    if st.sidebar.button("Download"):
        st.sidebar.download_button("Download Cleaned Text", cleaned_text)

# Sidebar Title
st.sidebar.text("Document Analysis App")
