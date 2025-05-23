from sentence_transformers import SentenceTransformer

# Load the model once (lightweight, fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_chunks):
    """
    Converts a list of text chunks into dense vector embeddings.
    """
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    return embeddings
