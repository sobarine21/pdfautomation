import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from docx import Document
from bs4 import BeautifulSoup
import numpy as np
import re
from wordcloud import WordCloud
from collections import Counter
from textstat import flesch_reading_ease, text_standard
from pptx import Presentation
from gtts import gTTS
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams

# Title and file uploader on the main page
st.title("Sobarine Content Analysis SaaS")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

# Function Definitions
def simple_sentiment_analysis(text):
    positive_words = set(["happy", "good", "great", "excellent", "positive", "wonderful", "amazing", "love"])
    negative_words = set(["sad", "bad", "terrible", "horrible", "negative", "hate", "awful"])
    
    score = 0
    for word in text.split():
        if word.lower() in positive_words:
            score += 1
        elif word.lower() in negative_words:
            score -= 1
    return score

def find_palindromes(text):
    words = text.split()
    palindromes = [word for word in words if word == word[::-1] and len(word) > 2]
    return palindromes

def display_palindromes(text):
    palindromes = find_palindromes(text)
    st.subheader("Palindromic Words")
    st.write(", ".join(palindromes) if palindromes else "No palindromic words found.")

def sentence_count(text):
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])  # Return count of non-empty sentences

def calculate_complexity(text):
    avg_sentence_length = len(text.split()) / sentence_count(text) if sentence_count(text) > 0 else 0
    avg_word_length = sum(len(word) for word in text.split()) / len(text.split()) if len(text.split()) > 0 else 0
    readability_score = flesch_reading_ease(text)
    return {
        "Average Sentence Length": avg_sentence_length,
        "Average Word Length": avg_word_length,
        "Readability Score": readability_score,
        "Text Standard": text_standard(text, float_output=True)
    }

def display_complexity(text):
    complexity = calculate_complexity(text)
    st.subheader("Text Complexity")
    st.json(complexity)

def mood_color(score):
    if score > 5:
        return "#A9DFBF"  # Light green for positive mood
    elif score < -5:
        return "#F5B7B1"  # Light red for negative mood
    else:
        return "#F9E79F"  # Light yellow for neutral mood

def display_text_with_mood_color(text):
    sentiment_score = simple_sentiment_analysis(text)
    color = mood_color(sentiment_score)
    st.markdown(f"<div style='background-color: {color}; padding: 10px;'>{text}</div>", unsafe_allow_html=True)

def estimate_reading_time(text, wpm=200):
    word_count = len(text.split())
    minutes = word_count / wpm
    return round(minutes, 2)

def display_reading_time(text):
    reading_time = estimate_reading_time(text)
    st.subheader("Estimated Reading Time")
    st.write(f"Approximate Reading Time: {reading_time} minutes")

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

def display_gendered_language(text):
    st.subheader("Gendered Language Analysis")
    result = gendered_language_analysis(text)
    st.write(result)

def sentiment_by_section(text):
    paragraphs = text.split("\n\n")
    section_sentiments = [simple_sentiment_analysis(paragraph) for paragraph in paragraphs]
    plt.bar(range(len(section_sentiments)), section_sentiments, color='orange')
    plt.title("Sentiment by Section")
    plt.xlabel("Section")
    plt.ylabel("Sentiment Score")
    st.pyplot()

def display_section_sentiment(text):
    st.subheader("Sentiment by Section")
    sentiment_by_section(text)

def find_jargon(text):
    common_words = set()  # Use a basic set instead of SpaCy stop words
    technical_words = set(word for word in text.split() if len(word) > 7 and word.lower() not in common_words)
    return technical_words

def display_jargon_finder(text):
    st.subheader("Technical Jargon")
    jargon = find_jargon(text)
    st.write(", ".join(jargon) if jargon else "No jargon found.")

def detect_tone(text):
    if "!" in text:
        return "Casual/Excited"
    elif text.isupper():
        return "Aggressive"
    else:
        return "Neutral/Formal"

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

def extract_emails(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails

def display_email_extraction(text):
    st.subheader("Email Addresses Found")
    emails = extract_emails(text)
    st.write(", ".join(emails) if emails else "No email addresses found.")

def highlight_text_in_pdf(pdf_file, keywords):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        for keyword in keywords:
            text_instances = page.search_for(keyword)
            for inst in text_instances:
                page.add_highlight_annot(inst)
    annotated_pdf_path = "annotated_pdf.pdf"
    doc.save(annotated_pdf_path)
    return annotated_pdf_path

def display_pdf_with_annotations(uploaded_file):
    st.subheader("Annotated PDF")
    keywords = st.text_input("Enter keywords to highlight in PDF (comma separated):", key=f"keywords_{uploaded_file.name}")
    if keywords:
        annotated_pdf = highlight_text_in_pdf(uploaded_file, keywords.split(","))
        st.download_button("Download Annotated PDF", annotated_pdf)

def analyze_style(text):
    formal_words = set(['therefore', 'consequently', 'moreover', 'hence', 'furthermore'])
    informal_words = set(['awesome', 'cool', 'totally', 'like', 'just', 'really'])
    formal_count = sum(1 for word in text.split() if word.lower() in formal_words)
    informal_count = sum(1 for word in text.split() if word.lower() in informal_words)
    return "Formal" if formal_count > informal_count else "Informal"

def display_style_analysis(text):
    style = analyze_style(text)
    st.subheader("Writing Style Analysis")
    st.write(f"Document Style: {style}")

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([text])
    indices = np.argsort(X.toarray()).flatten()[::-1]
    return [vectorizer.get_feature_names_out()[i] for i in indices[:top_n]]

def display_keyword_extraction(text):
    st.subheader("Keyword Extraction")
    keywords = extract_keywords(text)
    st.write(", ".join(keywords))

def ngram_analysis(text, n=2):
    words = text.split()
    n_grams = ngrams(words, n)
    n_gram_freq = Counter(n_grams)
    return n_gram_freq.most_common(10)

def display_ngram_analysis(text):
    st.subheader("N-gram Analysis (Bigrams)")
    ngram_counts = ngram_analysis(text, n=2)
    ngram_df = pd.DataFrame(ngram_counts, columns=["N-gram", "Frequency"])
    st.bar_chart(ngram_df.set_index("N-gram"))

def frequency_distribution(text):
    words = re.findall(r'\w+', text.lower())
    freq_dist = Counter(words)
    return freq_dist.most_common(10)

def display_frequency_distribution(text):
    st.subheader("Top Word Frequency Distribution")
    freq_dist = frequency_distribution(text)
    freq_df = pd.DataFrame(freq_dist, columns=["Word", "Frequency"])
    st.bar_chart(freq_df.set_index("Word"))

# Process each uploaded file
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing {uploaded_file.name}")

        # Read the contents of the uploaded file based on its type
        if uploaded_file.type == "application/pdf":
            # Read PDF file
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                st.text_area("Extracted Text from PDF", text, height=300)
                display_complexity(text)
                display_reading_time(text)
                display_section_sentiment(text)
                display_email_extraction(text)
                display_ngram_analysis(text)
                display_frequency_distribution(text)
                display_style_analysis(text)
                display_jargon_finder(text)
                display_gendered_language(text)
                display_palindromes(text)
                display_text_with_mood_color(text)
                display_keyword_extraction(text)
                display_pdf_with_annotations(uploaded_file)

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Read DOCX file
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            st.text_area("Extracted Text from DOCX", text, height=300)
            display_complexity(text)
            display_reading_time(text)
            display_section_sentiment(text)
            display_email_extraction(text)
            display_ngram_analysis(text)
            display_frequency_distribution(text)
            display_style_analysis(text)
            display_jargon_finder(text)
            display_gendered_language(text)
            display_palindromes(text)
            display_text_with_mood_color(text)
            display_keyword_extraction(text)

        elif uploaded_file.type == "text/plain":
            # Read TXT file
            text = uploaded_file.read().decode("utf-8")
            st.text_area("Extracted Text from TXT", text, height=300)
            display_complexity(text)
            display_reading_time(text)
            display_section_sentiment(text)
            display_email_extraction(text)
            display_ngram_analysis(text)
            display_frequency_distribution(text)
            display_style_analysis(text)
            display_jargon_finder(text)
            display_gendered_language(text)
            display_palindromes(text)
            display_text_with_mood_color(text)
            display_keyword_extraction(text)

        elif uploaded_file.type == "text/html":
            # Read HTML file
            soup = BeautifulSoup(uploaded_file.read(), 'html.parser')
            text = soup.get_text()
            st.text_area("Extracted Text from HTML", text, height=300)
            display_complexity(text)
            display_reading_time(text)
            display_section_sentiment(text)
            display_email_extraction(text)
            display_ngram_analysis(text)
            display_frequency_distribution(text)
            display_style_analysis(text)
            display_jargon_finder(text)
            display_gendered_language(text)
            display_palindromes(text)
            display_text_with_mood_color(text)
            display_keyword_extraction(text)

st.write("Analysis completed. You can upload more files to analyze!")
