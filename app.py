# app.py
import streamlit as st
from transformers import pipeline
from utils.pdf_parser import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# --- 1. CONFIGURATION AND MODEL LOADING ---
st.set_page_config(page_title="arXiv Recommender", layout="wide")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_faiss_index(index_file="data/faiss_index.bin", data_file="data/indexed_papers.json"):
    """Loads the pre-built FAISS index and the paper data."""
    try:
        index = faiss.read_index(index_file)
        with open(data_file, 'r', encoding='utf-8') as f:
            indexed_papers = json.load(f)
        return index, indexed_papers
    except FileNotFoundError:
        st.error(f"Required index files not found. Please run 'python build_index.py' first.")
        return None, None

summarizer = load_summarizer()
sentence_model = load_sentence_model()
faiss_index, indexed_papers = load_faiss_index()

# --- 2. RECOMMENDATION FUNCTION ---
def get_similar_papers_from_local(query_abstract, limit=5):
    """
    Finds similar papers from the local corpus using FAISS index.
    """
    if faiss_index is None or indexed_papers is None:
        return []
    
    query_embedding = sentence_model.encode([query_abstract], convert_to_tensor=False).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = faiss_index.search(query_embedding, limit)
    
    recommended_papers = [indexed_papers[i] for i in indices[0]]
    
    return recommended_papers

# --- 3. UI LAYOUT ---
st.title("📚 AI-Based arXiv Paper Recommender")
st.markdown("Instantly find relevant papers and get a concise summary from a local arXiv database.")

with st.sidebar:
    st.header("Input Options")
    input_method = st.radio("Choose Input Method", ["Upload PDF", "Enter Paper Abstract"])

    user_input_text = None
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF of a research paper", type=["pdf"])
        if uploaded_file:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            paper_details = extract_text_from_pdf("temp.pdf")
            if paper_details:
                user_input_text = paper_details['abstract']
                st.success("PDF uploaded and parsed successfully!")
                st.subheader(f"Recommendations for: **{paper_details['title']}**")
    
    elif input_method == "Enter Paper Abstract":
        abstract_input = st.text_area("Enter a paper abstract or research topic:", height=200)
        if st.button("Find Similar Papers"):
            if abstract_input:
                user_input_text = abstract_input
                st.subheader("Recommendations for your input:")

if user_input_text:
    with st.spinner("Finding similar papers..."):
        recommended_papers = get_similar_papers_from_local(user_input_text, limit=5)
    
    if recommended_papers:
        for i, paper in enumerate(recommended_papers):
            with st.expander(f"**{i+1}. {paper['title']}**"):
                authors_str = ', '.join(paper.get('authors', ['N/A']))
                st.markdown(f"**Authors:** {authors_str}")
                
                abstract = paper.get('abstract', 'No abstract available.')
                
                st.markdown("---")
                st.markdown("**Original Abstract:**")
                st.write(abstract)
                
                if abstract != 'No abstract available.':
                    st.markdown("---")
                    st.markdown("**Generated Summary:**")
                    try:
                        summary = summarizer(abstract, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                        st.info(summary)
                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}")
    else:
        if faiss_index is not None:
            st.warning("No papers found. Please try a different query.")