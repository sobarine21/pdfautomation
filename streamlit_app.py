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

# Set page configuration
st.set_page_config(page_title="Ultimate Document Analysis Platform", layout="wide")

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

# Number Extraction
def extract_numbers(text):
    return re.findall(r'\b\d+\b', text)

# Frequency Distribution of Words
def word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    return word_counts.most_common(10)

# Simple Word Tagging (Simulated)
def simple_pos_tagging(text):
    words = text.split()
    tagged = [(word, 'NN' if word[0].isupper() else 'NNP') for word in words]  # Basic tagging for demonstration
    return tagged

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
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    df_tfidf = pd.DataFrame(denselist, columns=feature_names)
    st.write(df_tfidf.T.sort_values(0, ascending=False).head(10))

    # Generate Word Cloud
    st.subheader("Keyword Cloud")
    generate_word_cloud(cleaned_text)

    # Email and Link Extraction
    st.subheader("Extracted Emails")
    st.write(extract_emails(cleaned_text))
    st.subheader("Extracted Links")
    st.write(extract_links(cleaned_text))
    st.subheader("Extracted Numbers")
    st.write(extract_numbers(cleaned_text))

    # Word Frequency Distribution
    st.subheader("Word Frequency Distribution")
    word_counts = word_frequency(cleaned_text)
    plot_word_frequency(word_counts)

    # Simple Word Tagging
    st.subheader("Simple Word Tagging")
    pos_tags = simple_pos_tagging(cleaned_text)
    st.write(pos_tags)

    # Stop Words Removal
    st.subheader("Text After Stop Words Removal")
    no_stopwords_text = remove_stopwords(cleaned_text)
    st.write(no_stopwords_text)

    # Visualizations
    st.subheader("Data Visualizations")
    plt.figure(figsize=(6, 4))
    sns.histplot([sentiment_score], bins=10, kde=True)
    plt.title("Sentiment Distribution")
    st.pyplot(plt)

    # Export Options
    st.subheader("Export Options")
    export_format = st.selectbox("Choose export format", ["CSV", "JSON", "Excel"])
    if st.button("Export Data"):
        if export_format == "CSV":
            output = df_tfidf.to_csv(index=False)
            st.download_button(label="Download CSV", data=output, file_name='keywords.csv', mime='text/csv')
        elif export_format == "JSON":
            output = df_tfidf.to_json()
            st.download_button(label="Download JSON", data=output, file_name='keywords.json', mime='application/json')
        elif export_format == "Excel":
            output = df_tfidf.to_excel(index=False)
            st.download_button(label="Download Excel", data=output, file_name='keywords.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # User Feedback Form
    st.sidebar.markdown("### Feedback")
    feedback = st.sidebar.text_area("Your Feedback:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")

# Help and Support Section
st.sidebar.markdown("### Help")
st.sidebar.write("For support, please contact us.")
