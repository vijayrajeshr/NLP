'''
This program is a Streamlit-based Web Application that performs Retrieval-Augmented Generation (RAG) LOGIC ; BUT NOT GENERATION!! using a PDF as its knowledge base. It allows users to upload a document and receive instant, mathematically-backed answers to their questions.

Functional Summary -------------------------------------

PDF Extraction: It uses PyPDF2 to scrape raw text from an uploaded file, cleaning up "noise" like line breaks to ensure words are read correctly.

Text Chunking: The code breaks the document into individual sentences so the search engine can pinpoint specific information rather than analyzing the whole file at once.

TF-IDF Vectorization: It converts human words into a numerical matrix. This allows the computer to understand which words are "important" (unique to the topic) and which are "filler" (like "the" or "is").

Cosine Similarity Search: When you ask a question, the program calculates the "mathematical distance" between your query and every sentence in the PDF to find the closest match.

Web Interface: Streamlit provides a clean, interactive front-end where the user interacts with text boxes and uploaders instead of a raw terminal.

Technical Stack----------------------------------------

Frontend: Streamlit (Python-based UI)

Parsing: PyPDF2 (PDF data extraction)

Logic/Math: Scikit-learn (TfidfVectorizer and cosine_similarity)
'''

import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Simple PDF Bot", layout="centered")

# --- UI HEADER ---
st.title(" Chat with your PDF")

st.caption("Upload a document and ask questions in plain English.")
st.caption("By Vijay Rajesh R")
st.divider() # Adds a clean thin line

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

def load_pdf(file):
    text_chunks = []
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            sentences = content.replace('\n', ' ').split('. ')
            text_chunks.extend(sentences)
    return [s.strip() for s in text_chunks if len(s) > 15]

if uploaded_file:
    sentences = load_pdf(uploaded_file)
    
    # --- CHAT INPUT ---
    user_query = st.text_input("What would you like to know?", placeholder="e.g. What is the main topic?")

    if user_query:
        # Math Logic
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(sentences + [user_query])
        similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
        best_match_index = similarity.argsort()[0][-1]
        best_score = similarity[0][best_match_index]

        # --- SIMPLE RESULT BOX ---
        st.write("### Answer:")
        if best_score > 0.1:
            st.success(f"Score: {best_score:.2f} | Answer: {sentences[best_match_index]}")
            
        else:
            st.error("Sorry, I couldn't find a relevant answer in that document.")
            
            