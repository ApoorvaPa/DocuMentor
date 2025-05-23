import faiss
import numpy as np
import os
from pathlib import Path
import pickle

VECTOR_DIR = Path("data/vectors")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

def save_faiss_index(vectors, chunks, index_name="doc_index"):
    """
    Saves vectors and their corresponding text chunks to FAISS and pickle.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(VECTOR_DIR / f"{index_name}.index"))

    # Save the chunks separately for later retrieval
    with open(VECTOR_DIR / f"{index_name}.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(index_name="doc_index"):
    """
    Loads the FAISS index and chunk metadata.
    """
    index = faiss.read_index(str(VECTOR_DIR / f"{index_name}.index"))
    with open(VECTOR_DIR / f"{index_name}.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search_index(query_embedding, top_k=3, index_name="doc_index"):
    """
    Searches the FAISS index for top_k similar chunks.
    """
    index, chunks = load_faiss_index(index_name)
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]
