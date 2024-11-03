import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
import numpy as np
from collections import Counter

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_data(text):
    lines = text.splitlines()
    data = [{"Line": line.strip()} for line in lines if line.strip()]
    return pd.DataFrame(data)

def search_keywords(text, keywords):
    results = {keyword: len(re.findall(keyword, text, re.IGNORECASE)) for keyword in keywords}
    return results

def summarize_text(text):
    return "\n".join(text.splitlines()[:5])

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

def count_words(text):
    return len(text.split())

def count_sentences(text):
    return len(re.split(r'[.!?]+', text)) - 1

def extract_links(text):
    return re.findall(r'https?://\S+', text)

def extract_emails(text):
    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

def find_phone_numbers(text):
    return re.findall(r'\+?\d[\d -]{8,12}\d', text)

def extract_dates(text):
    return re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)

def extract_tables(text):
    # Placeholder for table extraction logic
    return "Table extraction not implemented."

def count_paragraphs(text):
    return len([p for p in text.split('\n\n') if p.strip()])

def extract_unique_words(text):
    words = text.split()
    unique_words = set(words)
    return list(unique_words)

def frequency_distribution(text):
    words = text.split()
    return Counter(words).most_common(10)

def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = re.sub(f'({keyword})', r'**\1**', text, flags=re.IGNORECASE)
    return text

def generate_word_cloud(text):
    # Placeholder for word cloud generation
    return "Word cloud generation not implemented."

def create_summary_statistics(text):
    return {
        "Word Count": count_words(text),
        "Sentence Count": count_sentences(text),
        "Paragraph Count": count_paragraphs(text),
        "Unique Words": len(extract_unique_words(text))
    }

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def get_top_n_words(text, n=10):
    words = text.split()
    most_common = Counter(words).most_common(n)
    return dict(most_common)

def extract_n_grams(text, n=2):
    words = text.split()
    n_grams = zip(*[words[i:] for i in range(n)])
    return [' '.join(n_gram) for n_gram in n_grams]

def replace_text(text, old, new):
    return text.replace(old, new)

def split_text_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def calculate_readability(text):
    words = count_words(text)
    sentences = count_sentences(text)
    return 206.835 - (1.015 * (words / sentences)) - (84.6 * (len(text.split('\n')) / sentences))

# New Functions
def extract_headings(text):
    return [line for line in text.splitlines() if line.isupper() and line.strip()]

def count_images(pdf_file):
    with fitz.open(pdf_file) as doc:
        return sum(1 for page in doc for img in page.get_images(full=True))

def extract_image_metadata(pdf_file):
    images_metadata = []
    with fitz.open(pdf_file) as doc:
        for page in doc:
            for img in page.get_images(full=True):
                images_metadata.append({"page": page.number, "image_index": img[0], "image_xref": img[1]})
    return images_metadata

def find_repeated_phrases(text, phrase_length=3):
    words = text.split()
    phrases = [' '.join(words[i:i + phrase_length]) for i in range(len(words) - phrase_length + 1)]
    return Counter(phrases).most_common()

def extract_keywords_from_text(text):
    words = text.split()
    return Counter(words).most_common(20)

def generate_summary(text):
    return '. '.join(text.split('. ')[:3])

def replace_keywords(text, replacements):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_sentiment(text):
    return "Sentiment analysis not implemented."

def get_characters_count(text):
    return len(text)

def extract_code_blocks(text):
    return re.findall(r'```(.*?)```', text, re.DOTALL)

def count_words_by_length(text):
    word_lengths = Counter(len(word) for word in text.split())
    return dict(word_lengths)

def create_word_frequency_chart(word_freq):
    return "Word frequency chart generation not implemented."

def identify_references(text):
    return re.findall(r'\[(.*?)\]', text)

def extract_section_titles(text):
    return [line for line in text.splitlines() if line.startswith('#')]

def extract_all_tables(text):
    return "All tables extraction not implemented."

def count_unique_phrases(text, phrase_length=2):
    words = text.split()
    phrases = [' '.join(words[i:i + phrase_length]) for i in range(len(words) - phrase_length + 1)]
    return len(set(phrases))

def calculate_average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0

def convert_text_to_uppercase(text):
    return text.upper()

def find_acronyms(text):
    return re.findall(r'\b[A-Z]{2,}\b', text)

def extract_keywords_with_context(text, keywords):
    context_size = 10
    context = {}
    for keyword in keywords:
        matches = re.finditer(r'(\S+\s+){0,10}(' + re.escape(keyword) + r')(\s+\S+){0,10}', text, re.IGNORECASE)
        context[keyword] = [match.group(0) for match in matches]
    return context

def get_pdf_metadata(pdf_file):
    with fitz.open(pdf_file) as doc:
        return doc.metadata

def identify_duplicates(text):
    lines = text.splitlines()
    return [line for line, count in Counter(lines).items() if count > 1]

def get_word_lengths_distribution(text):
    word_lengths = [len(word) for word in text.split()]
    return Counter(word_lengths)

def validate_email_addresses(emails):
    return [email for email in emails if re.match(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email)]

# Streamlit UI
st.title("Comprehensive PDF Data Extraction SaaS")
st.write("Upload a PDF file to extract data.")

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
    stats = create_summary_statistics(text)
    st.write(stats)

    st.subheader("Extracted Links")
    st.write(extract_links(text))

    st.subheader("Extracted Emails")
    st.write(extract_emails(text))

    st.subheader("Extracted Phone Numbers")
    st.write(find_phone_numbers(text))

    st.subheader("Extracted Dates")
    st.write(extract_dates(text))

    st.subheader("Top N Words")
    top_words = get_top_n_words(text)
    st.write(top_words)

    st.subheader("Frequency Distribution of Words")
    freq_dist = frequency_distribution(text)
    st.write(freq_dist)

    st.subheader("Unique Words")
    st.write(extract_unique_words(text))

    st.subheader("Sentence Splitting")
    sentences = split_text_into_sentences(text)
    st.write(sentences)

    st.subheader("Readability Score")
    readability = calculate_readability(text)
    st.write(f"Readability Score: {readability:.2f}")

    st.subheader("Headings")
    st.write(extract_headings(text))

    st.subheader("Image Count")
    st.write(count_images(uploaded_file))

    st.subheader("Image Metadata")
    st.write(extract_image_metadata(uploaded_file))

    st.subheader("Repeated Phrases")
    st.write(find_repeated_phrases(text))

    st.subheader("Keyword Extraction")
    st.write(extract_keywords_from_text(text))

    st.subheader("Summary of Text")
    st.write(generate_summary(text))

    st.subheader("Text Replacements")
    replacements = st.text_input("Enter replacements (old:new, comma-separated):")
    if replacements:
        replacements_dict = dict(pair.split(':') for pair in replacements.split(','))
        replaced_text = replace_keywords(text, replacements_dict)
        st.write(replaced_text)

    st.subheader("Extracted Code Blocks")
    st.write(extract_code_blocks(text))

    st.subheader("Average Word Length")
    avg_word_length = calculate_average_word_length(text)
    st.write(f"Average Word Length: {avg_word_length:.2f}")

    csv = convert_df_to_csv(data_df)
    st.download_button(
        label="Download extracted data as CSV",
        data=csv,
        file_name='extracted_data.csv',
        mime='text/csv'
    )

    st.write(extract_tables(text))
    st.write(generate_word_cloud(text))
