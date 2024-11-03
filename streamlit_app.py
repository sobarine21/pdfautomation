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

# File uploader on the main page
uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

# Further processing can go here
if uploaded_files:
    for file in uploaded_files:
        st.write(f"Uploaded file: {file.name}")
        # Add your processing code for each file here

# Sidebar for file upload
st.sidebar.title("Upload Document")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

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

# Summarization
def summarize_text(text, num_sentences=3):
    sentences = text.split('. ')
    return '. '.join(sentences[:num_sentences])

# Simple sentiment analysis
def simple_sentiment_analysis(text):
    words = text.split()
    positive_words = set(['good', 'great', 'excellent', 'positive', 'fortunate', 'correct', 'superior'])
    negative_words = set(['bad', 'poor', 'terrible', 'negative', 'unfortunate', 'wrong', 'inferior'])
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    return positive_count - negative_count

# Keyword Cloud Generation
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Email Extraction
def extract_emails(text):
    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

# Link Extraction
def extract_links(text):
    return re.findall(r'https?://\S+', text)

# Frequency Distribution of Words
def word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    return word_counts.most_common(10)

# Stop Words Removal
def remove_stopwords(text):
    stop_words = set([
        'the', 'is', 'in', 'and', 'to', 'with', 'that', 'of', 'a', 'for', 'on', 
        'it', 'as', 'was', 'at', 'by', 'an', 'be', 'this', 'which', 'or', 'are'
    ])
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Visualizing Word Frequency
def plot_word_frequency(word_counts):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.xticks(rotation=45)
    plt.title('Top 10 Most Frequent Words')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Additional Features
def character_count(text):
    return len(text)

def word_count(text):
    return len(text.split())

def sentence_count(text):
    return text.count('.') + text.count('!') + text.count('?')

def longest_word(text):
    words = text.split()
    return max(words, key=len)

def shortest_word(text):
    words = text.split()
    return min(words, key=len)

def average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)

def top_n_words(text, n=10):
    word_counts = Counter(text.split())
    return word_counts.most_common(n)

def extract_hashtags(text):
    return re.findall(r'#\w+', text)

def extract_mentions(text):
    return re.findall(r'@\w+', text)

def top_n_sentences(text, n=3):
    sentences = text.split('. ')
    return sorted(sentences, key=len, reverse=True)[:n]

def find_unique_words(text):
    return set(text.split())

def common_phrases(text, n=5):
    phrases = re.findall(r'\b(\w+\s+\w+)\b', text)
    return Counter(phrases).most_common(n)

def keyword_extraction_tfidf(text):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=feature_names).T.sort_values(0, ascending=False).head(10)

def count_paragraphs(text):
    return text.count('\n') + 1

def check_spelling(text):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    return list(misspelled)

def extract_dates(text):
    return re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)

def extract_time(text):
    return re.findall(r'\b\d{1,2}:\d{2}\s*[AP]?[M]?\b', text)

def text_to_uppercase(text):
    return text.upper()

def text_to_lowercase(text):
    return text.lower()

def text_title_case(text):
    return text.title()

def remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

def extract_citations(text):
    return re.findall(r'\([^\)]+\)', text)

def extract_references(text):
    return re.findall(r'\[.*?\]', text)

def count_links(text):
    return len(extract_links(text))

def count_emails(text):
    return len(extract_emails(text))

def extract_quotes(text):
    return re.findall(r'"(.*?)"', text)

def find_most_common_char(text):
    return Counter(text).most_common(1)

def text_statistics(text):
    return {
        "Character Count": character_count(text),
        "Word Count": word_count(text),
        "Sentence Count": sentence_count(text),
        "Longest Word": longest_word(text),
        "Shortest Word": shortest_word(text),
        "Average Word Length": average_word_length(text)
    }

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
    st.write(f"Total Emails: {count_emails(cleaned_text)}")

    st.subheader("Extracted Links")
    links = extract_links(cleaned_text)
    st.write(links)
    st.write(f"Total Links: {count_links(cleaned_text)}")

    st.subheader("Extracted Numbers")
    st.write(extract_dates(cleaned_text))

    # Frequency Distribution
    st.subheader("Top 10 Most Frequent Words")
    freq_words = word_frequency(cleaned_text)
    plot_word_frequency(freq_words)

    # Unique Words
    st.subheader("Unique Words")
    st.write(find_unique_words(cleaned_text))

    # Quotes
    st.subheader("Extracted Quotes")
    quotes = extract_quotes(cleaned_text)
    st.write(quotes)

    # Check for spelling errors
    st.subheader("Spelling Errors")
    spelling_errors = check_spelling(cleaned_text)
    st.write(spelling_errors)

    # Paragraph Count
    st.subheader("Paragraph Count")
    st.write(count_paragraphs(cleaned_text))

    # Citations
    st.subheader("Extracted Citations")
    citations = extract_citations(cleaned_text)
    st.write(citations)

    # References
    st.subheader("Extracted References")
    references = extract_references(cleaned_text)
    st.write(references)

    # Special Character Removal
    st.subheader("Text without Special Characters")
    st.write(remove_special_characters(cleaned_text))

    # Save processed text if needed
    st.sidebar.subheader("Download Processed Text")
    if st.sidebar.button("Download"):
        st.sidebar.download_button("Download Cleaned Text", cleaned_text)

st.sidebar.text("Document Analysis App")
