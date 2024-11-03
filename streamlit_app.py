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
from pptx import Presentation
import spacy
from spellchecker import SpellChecker

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Title and file uploader on the main page
st.title("Document Analysis App")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

# Function to find palindromes in the text
def find_palindromes(text):
    words = text.split()
    palindromes = [word for word in words if word == word[::-1] and len(word) > 2]
    return palindromes

# Display palindromic words
def display_palindromes(text):
    palindromes = find_palindromes(text)
    st.subheader("Palindromic Words")
    st.write(", ".join(palindromes))

# Calculate text complexity metrics
def calculate_complexity(text):
    avg_sentence_length = len(text.split()) / sentence_count(text)
    avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
    readability_score = flesch_reading_ease(text)
    return {
        "Average Sentence Length": avg_sentence_length,
        "Average Word Length": avg_word_length,
        "Readability Score": readability_score
    }

# Display text complexity metrics
def display_complexity(text):
    complexity = calculate_complexity(text)
    st.subheader("Text Complexity")
    st.json(complexity)

# Determine mood color based on sentiment score
def mood_color(score):
    if score > 5:
        return "#A9DFBF"  # light green for positive mood
    elif score < -5:
        return "#F5B7B1"  # light red for negative mood
    else:
        return "#F9E79F"  # light yellow for neutral mood

# Display text with background color based on sentiment
def display_text_with_mood_color(text):
    sentiment_score = simple_sentiment_analysis(text)
    color = mood_color(sentiment_score)
    st.markdown(f"<div style='background-color: {color}; padding: 10px;'>{text}</div>", unsafe_allow_html=True)

# Estimate reading time for the text
def estimate_reading_time(text, wpm=200):
    word_count = len(text.split())
    minutes = word_count / wpm
    return round(minutes, 2)

# Display estimated reading time
def display_reading_time(text):
    reading_time = estimate_reading_time(text)
    st.subheader("Estimated Reading Time")
    st.write(f"Approximate Reading Time: {reading_time} minutes")

# Perform Part-of-Speech tagging
def pos_tagging(text):
    doc = nlp(text)
    pos_counts = Counter([token.pos_ for token in doc])
    return pos_counts

# Display Part-of-Speech analysis
def display_pos_tagging(text):
    st.subheader("Part-of-Speech Analysis")
    pos_counts = pos_tagging(text)
    st.bar_chart(pd.DataFrame(pos_counts.values(), index=pos_counts.keys(), columns=["Count"]))

# Analyze gendered language in the text
def gendered_language_analysis(text):
    male_words = ["he", "him", "his", "man", "men"]
    female_words = ["she", "her", "hers", "woman", "women"]
    
    male_count = sum(text.lower().count(word) for word in male_words)
    female_count = sum(text.lower().count(word) for word in female_words)
    
    if male_count > female_count:
        result = "More Male-Language Oriented"
    elif female_count > male_count:
        result = "More Female-Language Oriented"
    else:
        result = "Neutral Language"
    
    return result

# Display gendered language analysis result
def display_gendered_language(text):
    st.subheader("Gendered Language Analysis")
    result = gendered_language_analysis(text)
    st.write(result)

# Analyze sentiment by section
def sentiment_by_section(text):
    paragraphs = text.split("\n\n")
    section_sentiments = [simple_sentiment_analysis(paragraph) for paragraph in paragraphs]
    plt.bar(range(len(section_sentiments)), section_sentiments, color='orange')
    plt.title("Sentiment by Section")
    plt.xlabel("Section")
    plt.ylabel("Sentiment Score")
    st.pyplot(plt)

# Display section sentiment
def display_section_sentiment(text):
    st.subheader("Sentiment by Section")
    sentiment_by_section(text)

# Identify technical jargon in the text
def find_jargon(text):
    common_words = set(nlp.Defaults.stop_words)
    technical_words = set(word for word in text.split() if len(word) > 7 and word.lower() not in common_words)
    return technical_words

# Display identified jargon
def display_jargon_finder(text):
    st.subheader("Technical Jargon")
    jargon = find_jargon(text)
    st.write(", ".join(jargon))

# Detect tone of the text
def detect_tone(text):
    if "!" in text:
        return "Casual/Excited"
    elif text.isupper():
        return "Aggressive"
    else:
        return "Neutral/Formal"

# Create a PowerPoint presentation from the text
def create_presentation_from_text(text):
    prs = Presentation()
    slides = text.split('\n\n')
    for slide_content in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        title, content = slide_content[:30], slide_content[30:]  
        slide.shapes.title.text = title
        slide.placeholders[1].text = content
    prs.save("generated_presentation.pptx")
    return "generated_presentation.pptx"

# Highlight keywords in a PDF
def highlight_text_in_pdf(pdf_path, keywords):
    doc = fitz.open(pdf_path)
    for page in doc:
        for keyword in keywords:
            text_instances = page.search_for(keyword)
            for inst in text_instances:
                page.add_highlight_annot(inst)
    annotated_pdf_path = "annotated_" + pdf_path
    doc.save(annotated_pdf_path)
    return annotated_pdf_path

# Display annotated PDF with highlights
def display_pdf_with_annotations(pdf_path, keywords):
    st.subheader("Annotated PDF")
    annotated_pdf = highlight_text_in_pdf(pdf_path, keywords)
    st.download_button("Download Annotated PDF", annotated_pdf)

# Analyze the writing style
def analyze_style(text):
    formal_words = set(['therefore', 'consequently', 'moreover', 'hence', 'furthermore'])
    informal_words = set(['awesome', 'cool', 'totally', 'like', 'just', 'really'])
    formal_count = sum(1 for word in text.split() if word.lower() in formal_words)
    informal_count = sum(1 for word in text.split() if word.lower() in informal_words)
    return "Formal" if formal_count > informal_count else "Informal"

# Display writing style analysis
def display_style_analysis(text):
    style = analyze_style(text)
    st.subheader("Writing Style Analysis")
    st.write(f"Document Style: {style}")

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

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Extract text from HTML
def extract_text_from_html(file):
    soup = BeautifulSoup(file.read(), 'html.parser')
    return soup.get_text()

# Text Cleaning
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Summarization of text
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
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return list(set(emails))

# Main Analysis Loop
if uploaded_files:
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        cleaned_text = clean_text(text)
        
        st.subheader("Extracted Text")
        st.write(cleaned_text)

        display_palindromes(cleaned_text)
        display_complexity(cleaned_text)
        display_reading_time(cleaned_text)
        display_pos_tagging(cleaned_text)
        display_gendered_language(cleaned_text)
        display_jargon_finder(cleaned_text)
        display_style_analysis(cleaned_text)
        display_section_sentiment(cleaned_text)
        display_text_with_mood_color(cleaned_text)
        
        # Presentation Generation
        if st.button('Generate Presentation'):
            presentation_file = create_presentation_from_text(cleaned_text)
            st.success(f"Presentation created: {presentation_file}")
        
        # Word Cloud Generation
        generate_word_cloud(cleaned_text)

        # Email Extraction
        emails = extract_emails(cleaned_text)
        st.subheader("Extracted Emails")
        st.write(", ".join(emails))

        # PDF Annotation
        if uploaded_file.type == "application/pdf":
            keywords = st.text_input("Enter keywords to highlight in PDF:")
            if keywords:
                keywords_list = keywords.split(',')
                display_pdf_with_annotations(uploaded_file.name, keywords_list)

# To run the app, save this code to a file named 'app.py' and execute:
# streamlit run app.py
