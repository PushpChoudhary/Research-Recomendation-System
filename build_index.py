# build_index.py
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Define file paths
DATA_FILE_PATH = 'data/arxiv-metadata-oai-snapshot.json'
INDEX_FILE_PATH = 'data/faiss_index.bin'
INDEXED_PAPERS_PATH = 'data/indexed_papers.json'
MAX_PAPERS = 50000  # Set a limit to manage memory; adjust based on your system's specs.

def process_and_index_corpus():
    """
    Reads the arXiv dataset, creates embeddings for abstracts,
    and builds a FAISS index for efficient similarity search.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Raw data file '{DATA_FILE_PATH}' not found.")
        print("Please download the 'arxiv-metadata-oai-snapshot.json' from Kaggle and place it in the 'data' folder.")
        return

    print("Loading Sentence-Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Processing raw data from {DATA_FILE_PATH}...")
    corpus = []
    
    # The arXiv JSON file has one JSON object per line
    with open(DATA_FILE_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= MAX_PAPERS:
                break
            try:
                paper = json.loads(line)
                
                # --- FIX IS HERE ---
                # Correctly parse the authors_parsed field, which is a list of lists.
                authors = [' '.join(author_parts) for author_parts in paper.get('authors_parsed', [])]
                
                corpus.append({
                    'title': paper['title'].strip(),
                    'abstract': paper['abstract'].strip(),
                    'authors': authors
                })
            except json.JSONDecodeError as e:
                print(f"Skipping line {i} due to JSON parsing error: {e}")
            except Exception as e:
                print(f"Skipping line {i} due to other error: {e}")

    if not corpus:
        print("No papers were processed. Exiting.")
        return

    print(f"Generating embeddings for {len(corpus)} papers...")
    abstracts = [paper['abstract'] for paper in corpus]
    embeddings = model.encode(abstracts, convert_to_tensor=False)
    
    # Ensure embeddings are float32 for FAISS
    embeddings = embeddings.astype('float32')

    # Get embedding dimension
    d = embeddings.shape[1]
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(d)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add embeddings to the index
    index.add(embeddings)

    # Save the FAISS index to a file
    faiss.write_index(index, INDEX_FILE_PATH)
    print(f"FAISS index saved to '{INDEX_FILE_PATH}'")

    # Save the processed corpus data
    with open(INDEXED_PAPERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"Processed corpus data saved to '{INDEXED_PAPERS_PATH}'")

if __name__ == "__main__":
    process_and_index_corpus()