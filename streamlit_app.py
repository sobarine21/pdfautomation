import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import pytesseract  # OCR
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from bs4 import BeautifulSoup
import numpy as np
import re
import os
from wordcloud import WordCloud

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

# Entity Recognition (simple version)
def extract_entities(text):
    blob = TextBlob(text)
    return [(word, tag) for word, tag in blob.tags if tag in ['NNP', 'NNPS']]  # Proper nouns

# Keyword Cloud Generation
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
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

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    polarity = TextBlob(cleaned_text).sentiment.polarity
    st.write(f"Sentiment Polarity: {polarity:.2f}")

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

    # Entity Recognition
    st.subheader("Extracted Entities")
    entities = extract_entities(cleaned_text)
    st.write(entities)

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

    # Visualizations
    st.subheader("Data Visualizations")
    plt.figure(figsize=(6, 4))
    sns.histplot([polarity], bins=10, kde=True)
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
