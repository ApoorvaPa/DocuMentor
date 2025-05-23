# from fastapi import APIRouter, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import os
# from app.parser import parse_pdf, parse_image
# from pathlib import Path

# UPLOAD_DIR = Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)

# router = APIRouter()

# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     file_ext = Path(file.filename).suffix.lower()
#     if file_ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type")

#     file_path = UPLOAD_DIR / file.filename
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     if file_ext == ".pdf":
#         text = parse_pdf(file_path)
#     else:
#         text = parse_image(file_path)

#     return JSONResponse({"filename": file.filename, "extracted_text": text})  # preview only
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from app.parser import parse_pdf, parse_image
from app.utils import chunk_text
from app.embedder import get_embeddings
import os
from app.rag_core import save_faiss_index
from app.llm_loader import load_llm

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter()

@router.get("/ping")
async def ping():
    return {"status": "ok"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save file to /uploads
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Parse text from file
    if file_ext == ".pdf":
        text = parse_pdf(file_path)
    else:
        text = parse_image(file_path)

    # Chunk and embed
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    save_faiss_index(embeddings, chunks)

    return JSONResponse({
        "filename": file.filename,
        "extracted_text": text,
        "num_chunks": len(chunks),
        "embedding_shape": embeddings.shape,
        "preview_chunk": chunks[0][:500] if chunks else "No text extracted."
    })

llm_pipeline = load_llm()

@router.post("/query")
async def query_index(query: str = Form(...)):
    # Step 1: Embed the query
    query_embedding = get_embeddings([query])  # Shape: (1, 384)

    # Step 2: Search FAISS for relevant chunks
    top_chunks = search_index(query_embedding)

    # Step 3: Generate answer using LLM
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm_pipeline(prompt, max_new_tokens=150, do_sample=False)

    return {
        "query": query,
        "top_chunks": top_chunks,
        "answer": response[0]["generated_text"]
    }
