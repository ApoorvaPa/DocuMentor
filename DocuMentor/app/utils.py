def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits input text into overlapping word chunks.
    Each chunk is 'chunk_size' words, with 'overlap' between them.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        #Checking chunking from linux
    return chunks
